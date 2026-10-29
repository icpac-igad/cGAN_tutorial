"""STEP 4 -- realize one 0.1 deg IFS cycle over the East Africa box: 14 cGAN
fields x 51 members x 29 lead times, into a real (not virtual) Icechunk store.

Store layout (one store per cycle; README section 5):

    <field>   float32 (time=1, step=29, number=51, latitude=401, longitude=331)
              chunk = one (step) block, 27 MB raw; NaN where never written
    time, step (hours), number, latitude (25 -> -15), longitude (20 -> 53)
    root attrs: virtual-store spec + snapshot, box, read method, counts

Where it runs
-------------
  local cluster   --workers N --threads T     (default)  everything on this host;
                  sink may be local: (or ceph:)
  EWC cluster     --scheduler IP:PORT                    the Frisky workers of
                  step3_deploy_workers.py; sink must be ceph: (the workers cannot
                  see this host's disk)

Examples
--------
  # local only, no source.coop anywhere (virtual store built by step1 locally)
  python step4_realize.py --date 20261021 --vd local:$HOME/icechunk-local/ecmwf-ifs-0p1-icechunk-vd \\
      --sink local:$HOME/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261021T00z

  # only what the SEWAA products need (config.PRODUCTS), public virtual store, local sink
  python step4_realize.py --date 20261026 --product jurre-6h mvua-7d \\
      --sink local:$HOME/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261026T00z-jurre-6h+mvua-7d

  # EWC workers -> Ceph, reading the public virtual store
  python step4_realize.py --date 20261021 --sink ceph:cgan14-19e55e-25n15s/20261021T00z \\
      --env ~/.env.ceph --scheduler 192.168.1.164:8798
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

import cgan_dag as dag  # noqa: E402  -- strips AWS_* before any store opens
import config as C      # noqa: E402
import stores as S      # noqa: E402

import numpy as np      # noqa: E402


def create_schema(repo, fields, steps, members, lat, lon, time_val, attrs):
    """Empty arrays (NaN fill) + coordinates, metadata only, one commit."""
    import zarr
    sess = repo.writable_session("main")
    root = zarr.open_group(sess.store, mode="w", zarr_format=3, attributes=attrs)

    def coord(name, values, dtype, extra=None):
        a = root.create_array(name, shape=(len(values),), dtype=dtype,
                              dimension_names=[name], attributes=extra or {})
        a[:] = np.asarray(values, dtype=dtype)

    coord("time", [time_val], "int64", {"units": "hours since 1970-01-01",
                                        "calendar": "proleptic_gregorian"})
    coord("step", steps, "int32", {"units": "hours", "long_name": "lead time"})
    coord("number", members, "int32", {"long_name": "ensemble member (0 = control)"})
    coord("latitude", lat, "float64", {"units": "degrees_north"})
    coord("longitude", lon, "float64", {"units": "degrees_east"})
    shape = (1, len(steps), len(members), len(lat), len(lon))
    for name, (param, level, var) in fields.items():
        root.create_array(
            name, shape=shape, chunks=(1, 1) + shape[2:], dtype="float32",
            fill_value=float("nan"),
            dimension_names=["time", "step", "number", "latitude", "longitude"],
            attributes={"grib_param": param, "level_hPa": level,
                        "virtual_store_var": var,
                        "accumulated_from_step0": name in C.ACCUMULATED})
    return sess.commit(f"schema: {len(fields)} fields x {len(steps)} steps x "
                       f"{len(members)} members x {len(lat)}x{len(lon)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--run", default="00", choices=["00", "06", "12", "18"],
                    help="00/12 reach 168 h; 06/18 stop at 144 h (25 lead times)")
    ap.add_argument("--vd", default=C.vd_spec("sourcecoop"),
                    help="virtual store spec (coordinates, provenance, --full-read)")
    ap.add_argument("--sink", required=True, help="local:/path or ceph:<prefix>")
    ap.add_argument("--env", default=None, help="Ceph AK/SK file (ceph: sinks)")
    ap.add_argument("--scheduler", default=None,
                    help="EWC Frisky scheduler IP:PORT; omit for a local cluster")
    ap.add_argument("--workers", type=int, default=2, help="local cluster processes")
    ap.add_argument("--threads", type=int, default=4, help="local: block tasks per process")
    ap.add_argument("--pool", type=int, default=4, help="GET threads inside one block task")
    ap.add_argument("--max-inflight", type=int, default=40,
                    help="blocks queued at once (sliding window; bounds memory)")
    ap.add_argument("--fields", nargs="+", default=list(C.FIELDS))
    ap.add_argument("--steps", type=int, nargs="+", default=None,
                    help="lead times, h (default: config.STEPS the run has)")
    ap.add_argument("--product", nargs="+", choices=list(C.PRODUCTS), default=None,
                    help="realize only the lead times these SEWAA products need "
                         "(config.PRODUCTS; jurre-6h 30..54 h, mvua-7d 0..174 h)")
    ap.add_argument("--members", type=int, default=len(C.MEMBERS),
                    help="first N members (0 = control)")
    ap.add_argument("--full-read", action="store_true",
                    help="read whole messages through the virtual store instead "
                         "of the prefix read")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--pause", type=int, default=60)
    ap.add_argument("--recreate", action="store_true",
                    help="wipe and re-lay the schema of an existing sink")
    a = ap.parse_args()

    if a.scheduler and S.parse(a.sink)[0] == "local":
        sys.exit("EWC workers cannot write this host's disk: use a ceph: sink "
                 "with --scheduler, or drop --scheduler for a local cluster")
    if a.full_read and a.scheduler and S.parse(a.vd)[0] == "local":
        sys.exit("--full-read on EWC workers needs a sourcecoop: virtual store")
    unknown = set(a.fields) - set(C.FIELDS)
    if unknown:
        sys.exit(f"unknown fields {sorted(unknown)}; known {list(C.FIELDS)}")
    fields = {f: C.FIELDS[f] for f in a.fields}
    members = list(range(a.members))
    date8, run, box = a.date, a.run, C.BOX
    ny, nx = C.box_shape(box)

    # -- 1. the virtual store: does it hold this cycle, and on what grid? ----
    t0 = time.time()
    ds, vd_snap = S.open_virtual_dataset(a.vd, run)
    tval = np.datetime64(f"{date8[:4]}-{date8[4:6]}-{date8[6:]}T{run}:00")
    if tval not in ds.time.values:
        sys.exit(f"{date8} {run}z is not in the virtual store {a.vd} "
                 f"(run step1_build_virtual.py first)")
    vd_steps = set(int(s / np.timedelta64(1, "h")) for s in ds.step.values)
    if a.product:
        if a.steps is not None:
            sys.exit("give --steps or --product, not both")
        steps = sorted({s for p in a.product for s in C.PRODUCTS[p]["steps"]})
        lack = sorted(set(steps) - vd_steps)
        if lack:
            sys.exit(f"{run}z has no {lack} h (06z/18z stop at 144 h): "
                     f"{' '.join(a.product)} cannot be made from this run")
    elif a.steps is None:                  # 06z/18z: 0..144 h, 25 of the 29
        steps = [s for s in C.STEPS if s in vd_steps]
        if len(steps) < len(C.STEPS):
            print(f"NOTE     {run}z stops at {max(vd_steps)} h: {len(steps)} of the "
                  f"{len(C.STEPS)} cGAN lead times")
    else:
        steps = list(a.steps)
    if set(steps) - vd_steps:
        sys.exit(f"steps {sorted(set(steps) - vd_steps)} not in the store")
    sub = ds["t2m"].sel(latitude=slice(box["lat_max"], box["lat_min"]),
                        longitude=slice(box["lon_min"], box["lon_max"]))
    lat, lon = sub.latitude.values, sub.longitude.values
    if (len(lat), len(lon)) != (ny, nx):
        sys.exit(f"box gives {len(lat)}x{len(lon)} on the store grid, expected {ny}x{nx}")
    print(f"virtual  {a.vd}  snapshot {vd_snap}  ({time.time() - t0:.1f}s)")
    print(f"box      lat {lat[0]:.1f}..{lat[-1]:.1f}  lon {lon[0]:.1f}..{lon[-1]:.1f}  "
          f"{ny} x {nx} at {box['deg']} deg")
    n_blocks = len(fields) * len(steps)
    print(f"work     {len(fields)} fields x {len(steps)} steps = {n_blocks} blocks x "
          f"{len(members)} members = {n_blocks * len(members):,} messages, "
          f"{'whole messages via the virtual store' if a.full_read else 'prefix read'}")

    # -- 2. the sink ---------------------------------------------------------
    repo, created = S.open_realized(a.sink, a.env, create=True)
    if created or a.recreate:
        attrs = dict(title="ECMWF IFS ENS 0.1 deg, realized for cGAN inference",
                     cycle=f"{date8} {run}z", box=box, fields=list(fields),
                     virtual_store=a.vd, virtual_store_snapshot=vd_snap,
                     read_method="virtual" if a.full_read else "prefix+libaec",
                     products=a.product or [],
                     source="s3://ecmwf-forecasts (fallback gs://ecmwf-open-data)",
                     workflow="cGAN_tutorial/ifs-0p1-gik")
        tv = int((tval - np.datetime64("1970-01-01T00:00")) / np.timedelta64(1, "h"))
        snap = create_schema(repo, fields, steps, members, lat, lon, tv, attrs)
        print(f"sink     {a.sink}  {'created' if created else 'recreated'}  schema {snap}")
    else:
        print(f"sink     {a.sink}  exists -- refilling all blocks (use --recreate "
              f"to change its shape)")

    # -- 3. the cluster ------------------------------------------------------
    import frisky
    cluster = None
    if a.scheduler:
        client = frisky.Client(a.scheduler)
        where = f"EWC scheduler {a.scheduler}"
    else:
        # get_client(), not Client(cluster): the latter raises, and the live
        # worker processes then keep the interpreter from exiting.
        cluster = frisky.LocalCluster(n_workers=a.workers, threads_per_worker=a.threads,
                                      processes=True, silence_summary=True)
        client = cluster.get_client()
        where = f"local cluster {a.workers} x {a.threads}"
    print(f"cluster  {where}, {a.pool} GETs per block task, "
          f"{a.max_inflight} blocks in flight\n", flush=True)
    try:
        run_blocks(a, repo, client, fields, steps, members, date8, run, box, vd_snap)
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


def run_blocks(a, repo, client, fields, steps, members, date8, run, box, vd_snap):
    import frisky
    n_blocks = len(fields) * len(steps)

    # -- 4. one fork, blocks through a sliding window, one commit -----------
    session = repo.writable_session("main")
    fork = session.fork()
    pending = [(name, si, step_h) for name in fields for si, step_h in enumerate(steps)]
    forks, tot, t_run = [], {}, time.time()
    done = 0
    for rnd in range(a.rounds):
        queue, failed, errs = list(pending), [], []
        meta, ac = {}, frisky.as_completed()

        def submit(spec):
            name, si, step_h = spec
            f = client.submit(dag.read_write_block, fork, name, si, fields[name], date8,
                              run, members, step_h, box, a.pool, a.vd, a.full_read)
            meta[f.key] = (spec, f)
            ac.add(f)

        for spec in queue[:a.max_inflight]:
            submit(spec)
        queue = queue[a.max_inflight:]
        while not ac.is_empty():
            f = next(ac)
            spec, f = meta.pop(f.key)
            try:
                fk, cnt = f.result()
                forks.append(fk)
                for k, v in cnt.items():
                    tot[k] = tot.get(k, 0) + v
                done += 1
            except Exception as exc:                          # noqa: BLE001
                failed.append(spec)
                errs.append(f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
            if queue:
                submit(queue.pop(0))
            if done and done % max(1, n_blocks // 10) == 0 and not failed:
                el = time.time() - t_run
                print(f"  {done:5d}/{n_blocks} blocks  {el:6.0f}s  "
                      f"{done * len(members) / el:5.1f} msg/s", flush=True)
        if not failed:
            break
        print(f"  round {rnd + 1}: {len(failed)} block(s) failed, e.g. {errs[0]}")
        pending = failed
        if rnd < a.rounds - 1:
            time.sleep(a.pause)
    else:
        sys.exit(f"{len(pending)} block(s) still failing after {a.rounds} rounds; "
                 f"nothing committed")

    session.merge(*forks)
    wall = time.time() - t_run
    n_msg = n_blocks * len(members)
    summary = dict(blocks=n_blocks, messages=n_msg, wall_s=round(wall, 1),
                   msg_per_s=round(n_msg / wall, 2))
    if tot.get("bytes_full"):
        summary.update(bytes_fetched=int(tot["bytes_fetched"]),
                       bytes_whole_messages=int(tot["bytes_full"]),
                       saved_pct=round(100 - 100 * tot["bytes_fetched"] / tot["bytes_full"], 1),
                       topup_gets=int(tot.get("topups", 0)))
    snap = session.commit(f"{date8} {run}z: {n_blocks} blocks, {n_msg} messages "
                          f"from virtual store snapshot {vd_snap}")
    import zarr
    s2 = repo.writable_session("main")
    zarr.open_group(s2.store, mode="r+").attrs.update({"realize_summary": summary})
    snap = s2.commit("realize summary")
    print(f"\ncommitted {snap}")
    print(f"wall     {wall:.0f}s  {n_msg / wall:.1f} msg/s  ({n_blocks} blocks)")
    if tot.get("bytes_full"):
        print(f"bytes    fetched {tot['bytes_fetched'] / 1e9:.2f} of "
              f"{tot['bytes_full'] / 1e9:.2f} GB whole-message "
              f"(saved {summary['saved_pct']}%), {summary['topup_gets']} top-up GETs, "
              f"mean GET {tot['t_get'] / tot['msgs']:.2f}s, decode "
              f"{1000 * tot['t_decode'] / tot['msgs']:.0f} ms")
    print(f"blocks   mean read {tot['t_read_block'] / n_blocks:.1f}s, "
          f"mean write {tot['t_write_block'] / n_blocks:.2f}s")


if __name__ == "__main__":
    main()
