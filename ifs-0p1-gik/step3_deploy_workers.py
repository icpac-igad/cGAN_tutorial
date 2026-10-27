"""STEP 3 -- stand up standalone Frisky workers on the EWC VMs, via the Dask cluster.

Only for EWC mode (README section 4); the local mode needs none of this.
Adapted from grib-index-kerchunk ecmwf/0p1/wide-box/realize/deploy_workers.py.
Everything this script starts lives under /tmp/frisky-cgan-ea on each VM and
uses scheduler port 8798 (dashboard 8793), so it never touches the CRMA daily
routine's workers (/tmp/frisky-ea, :8796) or any other copy.

Original notes:

Why standalone rather than `frisky.hijack(dask_client)`:

  * hijack runs a Frisky worker INSIDE each Dask worker process, so it inherits
    that process's environment -- including the Ceph `AWS_*` that makes the
    icechunk virtual-chunk client resolve a hostname that does not exist
    (README.md §2).  Clearing it would strip the Dask workers' credentials for
    every other job on the cluster, and README §2 rule 1 says putting it back
    mid-read does not work because the config is cached process-wide.
  * hijack also inherits `worker.state.nthreads` (hijack.py:605), which is 4.
    That is fine now -- see --nthreads below -- but it is not a choice.
  * hijack replaces the Dask dashboard at `/`.

A separate process per VM has its own environment, leaves the Dask workers
completely alone, and is torn down with `--stop`.

The Dask cluster is used only as a remote-exec channel: there is no SSH key on
the gateway, and `client.run()` reaches all six VMs.

On --nthreads 4: `NEXT_SESSION.md` §4.2 says 1, because Dask could not see this
workload's memory and over-subscribed until the nanny killed workers.  That was
a property of the EAGER task shape.  One task here holds one GRIB message
(~4 MB decoded) and returns ~0.1 MB, so 4 threads is ~20 MB of concurrent
working set against a 10 GB limit.  The reason for 1 does not apply, and 4
gives 24 concurrent readers across the six VMs.

Usage
-----
    P=.venv/bin/python
    $P deploy_workers.py --install     # build the venv on all six (~3 min)
    $P deploy_workers.py --start       # launch the workers
    $P deploy_workers.py --status
    $P deploy_workers.py --stop        # kill them; nothing else is touched
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Its own directory and venv, so the daily routine's /tmp/frisky-ea (venv,
# module, pid file) is never touched; --stop kills only workers started here.
REMOTE_DIR = "/tmp/frisky-cgan-ea"
PORT, DASHBOARD_PORT = 8798, 8793
MODULE = "cgan_dag.py"          # the only file the workers need
REMOTE_PY = f"{REMOTE_DIR}/.venv/bin/python"
REMOTE_FRISKY = f"{REMOTE_DIR}/.venv/bin/frisky"
DASK_SCHEDULER = "tcp://127.0.0.1:8786"

# Quoted individually when written into the install script -- an unquoted
# `frisky>=0.3.0` is a shell redirect, not a version specifier.
PACKAGES = ["frisky==0.7.2", "icechunk==2.1.1", "zarr==3.2.1",
            "xarray==2026.7.0", "gribberish==1.6.0", "numpy==2.5.1",
            "imagecodecs==2026.10.10"]   # libaec for the prefix decode


# ── remote functions.  Each returns fast; nothing blocks the Dask worker's
#    event loop for more than a moment, so heartbeats keep flowing. ──────────

def _kick_off_install(dirname, base_python, packages):
    """Start the venv build detached and return immediately."""
    import os, shlex, subprocess, socket
    os.makedirs(dirname, exist_ok=True)
    specs = " ".join(shlex.quote(s) for s in packages)
    script = f"""#!/bin/bash
set -e
rm -f {dirname}/INSTALL_OK {dirname}/INSTALL_FAIL
{base_python} -m venv {dirname}/.venv
{dirname}/.venv/bin/pip install -q --upgrade pip
{dirname}/.venv/bin/pip install -q {specs}
touch {dirname}/INSTALL_OK
"""
    path = f"{dirname}/install.sh"
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)
    log = open(f"{dirname}/install.log", "wb")
    subprocess.Popen(["/bin/bash", "-c",
                      f"{path} || touch {dirname}/INSTALL_FAIL"],
                     stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    return {"host": socket.gethostname(), "started": True}


def _check_install(dirname):
    import os, socket
    ok = os.path.exists(f"{dirname}/INSTALL_OK")
    fail = os.path.exists(f"{dirname}/INSTALL_FAIL")
    tail = ""
    if fail:
        try:
            with open(f"{dirname}/install.log") as f:
                tail = f.read()[-400:]
        except Exception:
            pass
    return {"host": socket.gethostname(),
            "state": "ok" if ok else ("FAILED" if fail else "building"),
            "tail": tail}


def _push_module(dirname, name, source):
    """Frisky pickles task functions BY REFERENCE, so the module defining them
    must be importable on the worker.  Ship it rather than rely on a shared
    filesystem."""
    import os, socket
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, name), "w") as f:
        f.write(source)
    return {"host": socket.gethostname(), "wrote": name, "bytes": len(source)}


def _start_worker(dirname, frisky_bin, sched, nthreads, memory_limit,
                  n_per_vm=1):
    """Launch `frisky worker` detached, with AWS_* scrubbed from ITS env only.

    The Dask worker process keeps its own AWS_* untouched -- this is a child
    with a modified copy, so no other job on this VM is affected.
    """
    import os, subprocess, socket
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env["PYTHONPATH"] = dirname
    env["CGAN_DAG_HOME"] = dirname
    log = open(f"{dirname}/worker.log", "ab")
    pids = []
    for _ in range(n_per_vm):
        p = subprocess.Popen(
            [frisky_bin, "worker", sched, "--nthreads", str(nthreads),
             "--memory-limit", memory_limit],
            env=env, cwd=dirname, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        pids.append(p.pid)
    with open(f"{dirname}/worker.pid", "w") as f:
        f.write("\n".join(str(x) for x in pids))
    return {"host": socket.gethostname(), "pids": pids}


def _start_scheduler(dirname, frisky_bin, bind, dashboard, tracing_capacity):
    """Run the Frisky scheduler on a worker VM, not the gateway.

    The gateway is the JupyterHub session and is capped at 8 GiB
    (BLOCKERS.md). A scheduler left up across several 82,710-task runs grew
    past that and was OOM-killed mid-run -- the client saw only
    `RuntimeError: Failed to send: channel closed`, and the store was left
    with nothing but its initial commit. A worker VM has 15 GB and no cgroup
    cap, and FRISKY_TRACING_CAPACITY bounds the span buffer that does most of
    the growing.
    """
    import os, subprocess, socket
    subprocess.run(["pkill", "-f", f"{dirname}/.venv/bin/frisky scheduler"],
                   capture_output=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env["FRISKY_TRACING_CAPACITY"] = str(tracing_capacity)
    log = open(f"{dirname}/scheduler.log", "ab")
    p = subprocess.Popen(
        [frisky_bin, "scheduler", "--address", bind,
         "--dashboard-address", dashboard],
        env=env, cwd=dirname, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    with open(f"{dirname}/scheduler.pid", "w") as f:
        f.write(str(p.pid))
    return {"host": socket.gethostname(), "pid": p.pid, "bind": bind}


def _stop_scheduler(dirname):
    import socket, subprocess
    r = subprocess.run(["pkill", "-f",
                        f"{dirname}/.venv/bin/frisky scheduler"],
                       capture_output=True)
    return {"host": socket.gethostname(), "pkill_rc": r.returncode}


def _worker_status(dirname):
    """What is actually running on this VM.

    `worker.pid` holds ONE line per launched process, so the old
    `int(f.read().strip())` raised ValueError the moment --workers-per-vm was
    2 -- and the bare `except` swallowed it, reporting `alive: False` for a VM
    whose workers were healthy and executing tasks.  A status check that says
    "dead" for a live cluster is worse than no status check.

    `pgrep` is the authoritative count, not the pid file: --start records only
    the most recent launch, so orphans from an earlier one are invisible to it
    (the same reason `_stop_worker` matches on the command line).  A `running`
    that exceeds `alive` means orphans -- and an orphan runs whatever version
    of the task module it started with, which is the stale-module trap.
    """
    import os, socket, subprocess

    def _count(pattern):
        r = subprocess.run(["pgrep", "-fc", pattern],
                           capture_output=True, text=True)
        try:
            return int(r.stdout.strip())
        except ValueError:
            return 0

    pids = []
    try:
        with open(f"{dirname}/worker.pid") as f:
            pids = [int(x) for x in f.read().split() if x.strip()]
    except Exception:
        pass

    alive = 0
    for pid in pids:
        try:
            os.kill(pid, 0)
            alive += 1
        except Exception:
            pass

    running = _count(f"{dirname}/.venv/bin/frisky worker")
    sched = _count(f"{dirname}/.venv/bin/frisky scheduler")

    # worker.log is mostly multi-KB JSON state dumps; slicing the last 300
    # bytes of one is unreadable noise.  Show the last human-readable lines.
    tail = []
    try:
        with open(f"{dirname}/worker.log") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if line and not line.startswith("{") and len(line) < 200:
                    tail.append(line)
    except Exception:
        pass

    return {"host": socket.gethostname(),
            "alive": f"{alive}/{len(pids)}" if pids else "0/0",
            "running": running, "orphans": max(running - alive, 0),
            "sched": sched, "venv": os.path.exists(
                f"{dirname}/.venv/bin/frisky"),
            "log": " | ".join(tail[-2:])}


def _stop_worker(dirname):
    """Kill EVERY frisky worker on this VM, not just the pid we last recorded.

    worker.pid holds only the most recent launch, so a --start that did not
    stop first leaves orphans the pid file can no longer name.  That is not
    cosmetic: an orphan runs whatever version of the task module it was
    started with, and Frisky pickles task functions by reference -- so a stale
    worker fails every task calling a function added since.  Match on the
    command line instead.
    """
    import socket, subprocess
    r = subprocess.run(["pkill", "-f", f"{dirname}/.venv/bin/frisky worker"],
                       capture_output=True, text=True)
    left = subprocess.run(["pgrep", "-fc",
                           f"{dirname}/.venv/bin/frisky worker"],
                          capture_output=True, text=True).stdout.strip()
    return {"host": socket.gethostname(), "pkill_rc": r.returncode,
            "still_running": left or "0"}


# ── client side ────────────────────────────────────────────────────────────

def show(label, res):
    print(f"\n{label}")
    for addr, r in sorted(res.items(), key=lambda kv: str(kv[1].get("host"))):
        host = r.pop("host", addr)
        print(f"  {host:16s} {r}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--install", action="store_true")
    p.add_argument("--scheduler-on", metavar="IP",
                   help="worker IP to host the Frisky scheduler, e.g. "
                        "192.168.1.74.  Keeps it off the 8 GiB gateway")
    p.add_argument("--start", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--tracing-capacity", type=int, default=20000,
                   help="scheduler span buffer; the default 200k is what "
                        "grew unbounded across runs")
    p.add_argument("--scheduler", default=f"192.168.1.164:{PORT}",
                   help="the FRISKY scheduler the workers should dial")
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--workers-per-vm", type=int, default=1,
                   help="Frisky worker PROCESSES per VM.  icechunk caps "
                        "in-flight connections per process: 1 proc tops out "
                        "at ~25 MB/s, 2 reach ~47, and past 2 it is flat. "
                        "Threads cannot substitute -- see procs_probe.py")
    p.add_argument("--memory-limit", default="10GB")
    p.add_argument("--dask", default=DASK_SCHEDULER)
    p.add_argument("--timeout", type=int, default=900,
                   help="seconds to wait for the venv build")
    args = p.parse_args()

    if not any([args.install, args.start, args.status, args.stop,
                args.scheduler_on]):
        p.error("pick one of --install / --scheduler-on / --start / "
                "--status / --stop")

    from distributed import Client

    with Client(args.dask, timeout=30) as c:
        if args.install:
            base = "/opt/mamba/envs/dask/bin/python3.12"
            show("kicking off venv builds",
                 c.run(_kick_off_install, REMOTE_DIR, base, PACKAGES))
            t0 = time.time()
            while time.time() - t0 < args.timeout:
                time.sleep(20)
                res = c.run(_check_install, REMOTE_DIR)
                states = [r["state"] for r in res.values()]
                done = sum(s == "ok" for s in states)
                print(f"  {time.time() - t0:5.0f}s  ok={done}/{len(states)}  "
                      f"{sorted(set(states))}")
                if all(s == "ok" for s in states):
                    break
                if any(s == "FAILED" for s in states):
                    show("FAILED", res)
                    sys.exit(1)
            else:
                show("timed out", c.run(_check_install, REMOTE_DIR))
                sys.exit(1)
            print("\nvenv ready on all workers")


        if args.scheduler_on:
            host = args.scheduler_on
            target = [a for a in c.scheduler_info()["workers"]
                      if f"//{host}:" in a]
            if not target:
                p.error(f"no Dask worker at {host}; known: "
                        f"{list(c.scheduler_info()['workers'])}")
            bind = f"{host}:{PORT}"
            show(f"starting frisky scheduler on {host}",
                 c.run(_start_scheduler, REMOTE_DIR, REMOTE_FRISKY, bind,
                       f"{host}:{DASHBOARD_PORT}", args.tracing_capacity,
                       workers=target[:1]))
            args.scheduler = bind
            print(f"\nscheduler  {bind}   dashboard http://{host}:{DASHBOARD_PORT}")
            print(f"tracing capacity {args.tracing_capacity} "
                  f"(default 200k is what grew unbounded)")

        if args.install or args.start:
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, MODULE)) as f:
                src = f.read()
            show(f"shipping {MODULE}",
                 c.run(_push_module, REMOTE_DIR, MODULE, src))

        if args.start:
            # Always clear first.  Starting on top of a running worker leaves
            # an orphan holding a stale copy of the task module.
            show("clearing any running workers",
                 c.run(_stop_worker, REMOTE_DIR))
            time.sleep(3)
            show("starting frisky workers",
                 c.run(_start_worker, REMOTE_DIR, REMOTE_FRISKY,
                       args.scheduler, args.nthreads, args.memory_limit,
                       args.workers_per_vm))
            time.sleep(5)
            show("status", c.run(_worker_status, REMOTE_DIR))
            n = len(c.scheduler_info()["workers"]) * args.workers_per_vm
            print(f"\n{n} frisky workers ({args.workers_per_vm}/VM) dialling "
                  f"{args.scheduler} at {args.nthreads} threads each "
                  f"({n * args.nthreads} concurrent readers)")

        if args.status:
            res = c.run(_worker_status, REMOTE_DIR)
            # Totalled BEFORE show(), which pops "host" out of each dict.
            running = sum(r["running"] for r in res.values())
            orphans = sum(r["orphans"] for r in res.values())
            sched = sum(r["sched"] for r in res.values())
            show("status", res)
            print(f"\n{running} frisky worker process(es) across "
                  f"{len(res)} VMs, {sched} scheduler(s)")
            if orphans:
                print(f"WARNING: {orphans} orphan(s) -- not in any pid file, "
                      f"so --start never stopped them.  An orphan runs the "
                      f"task module it was\n         launched with, and "
                      f"Frisky pickles task functions by reference.  "
                      f"Run --stop, then --start.")
            if not running:
                print("no workers running -- ./deploy_workers.py "
                      "--scheduler-on <ip> --start")
            elif not sched:
                print("workers are up but NO scheduler is running on these "
                      "VMs; they cannot be dialling one here.")

        if args.stop:
            show("stopping", c.run(_stop_worker, REMOTE_DIR))
            # Its scheduler goes too; others run from other dirs and are
            # not matched.
            show("stopping scheduler", c.run(_stop_scheduler, REMOTE_DIR))
            print("\nDask workers untouched -- their AWS_* was never modified")


if __name__ == "__main__":
    main()
