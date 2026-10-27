"""STEP 5 -- check a realized cycle before anything consumes it.

  1. complete   every (field, step) chunk is in the manifest -- an unwritten
                chunk reads as NaN, not as an error, so this must be asked
  2. exact      --samples random (field, step, member) boxes equal the same box
                read as a WHOLE message through the virtual store and decoded by
                gribberish (the prefix read must be bit-exact, not close)
  3. sane       all values finite; members differ; accumulated fields (tp, cp,
                ssr) are 0 at step 0 and never decrease with lead time

Exit status 0 only if all three pass.

  python step5_check.py --store local:$HOME/icechunk-local/.../20261021T00z \\
      --vd local:$HOME/icechunk-local/ecmwf-ifs-0p1-icechunk-vd
  python step5_check.py --store ceph:cgan14-19e55e-25n15s/20261021T00z --env ~/.env.ceph
  python step5_check.py --store sourcecoop:forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261021T00z
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cgan_dag as dag  # noqa: E402  -- strips AWS_*
import config as C      # noqa: E402
import stores as S      # noqa: E402

import numpy as np      # noqa: E402


async def _written(session, name):
    return {c async for c in session.chunk_coordinates(f"/{name}")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--vd", default=C.vd_spec("sourcecoop"))
    ap.add_argument("--env", default=None)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    repo, _ = S.open_realized(a.store, a.env)
    sess = repo.readonly_session("main")
    import xarray as xr
    ds = xr.open_zarr(sess.store, consolidated=False, zarr_format=3, decode_timedelta=True)
    fields = [f for f in ds.attrs.get("fields", C.FIELDS) if f in ds]
    steps = [int(s / np.timedelta64(1, "h")) for s in ds.step.values]
    members = [int(m) for m in ds.number.values]
    date8, run = ds.attrs["cycle"].split()[0], ds.attrs["cycle"].split()[1][:2]
    print(f"store    {a.store}  snapshot {sess.snapshot_id}")
    print(f"cycle    {date8} {run}z  {len(fields)} fields x {len(steps)} steps x "
          f"{len(members)} members x {ds.sizes['latitude']}x{ds.sizes['longitude']}")
    print(f"built    from {ds.attrs.get('virtual_store')} @ "
          f"{ds.attrs.get('virtual_store_snapshot')}, {ds.attrs.get('read_method')}")
    if "realize_summary" in ds.attrs:
        print(f"summary  {ds.attrs['realize_summary']}")
    ok = True

    # 1. complete -----------------------------------------------------------
    missing = {}
    for f in fields:
        got = asyncio.run(_written(sess, f))
        want = {(0, si, 0, 0, 0) for si in range(len(steps))}
        if want - got:
            missing[f] = sorted(c[1] for c in want - got)
    print(f"\n1 complete   {'yes' if not missing else 'NO'} -- "
          f"{len(fields) * len(steps) - sum(map(len, missing.values()))}/"
          f"{len(fields) * len(steps)} chunks written"
          + (f"; missing step indices {missing}" if missing else ""))
    ok &= not missing

    # 2. exact against the virtual store ------------------------------------
    rng = random.Random(a.seed)
    picks = [(rng.choice(fields), rng.randrange(len(steps)), rng.choice(members))
             for _ in range(a.samples)]
    n_eq = 0
    print(f"\n2 exact      {a.samples} boxes vs whole messages via {a.vd}")
    for f, si, m in picks:
        param, level, var = C.FIELDS[f]
        mine = ds[f].isel(time=0, step=si).sel(number=m).values
        ref = dag.read_virtual(a.vd, var, level, date8, run, m, steps[si], C.BOX)
        eq = np.array_equal(mine, ref, equal_nan=True)
        n_eq += eq
        print(f"     {f:5s} +{steps[si]:3d}h m{m:<2d}  {'equal' if eq else 'DIFF max |d| = %g' % np.nanmax(np.abs(mine - ref))}")
    print(f"   {n_eq}/{a.samples} bit-exact")
    ok &= n_eq == a.samples

    # 3. sane -----------------------------------------------------------------
    print("\n3 sane")
    bad = []
    for f in fields:
        x = ds[f].isel(time=0)
        means = x.mean(("number", "latitude", "longitude")).values
        finite = float(np.isfinite(x.isel(step=-1).values).mean())
        spread = float(x.isel(step=-1).std("number").mean().values)
        line = (f"     {f:5s} finite {finite:.3f}  spread@{steps[-1]}h {spread:10.4g}  "
                f"mean {means[0]:10.4g} -> {means[-1]:10.4g}")
        if finite < 1.0:
            bad.append(f"{f} not finite")
        if len(members) > 1 and spread == 0 and f != "mcc":
            bad.append(f"{f} members identical")
        if f in C.ACCUMULATED:
            if steps[0] == 0 and abs(means[0]) > 0:
                bad.append(f"{f} not 0 at step 0")
            if np.any(np.diff(means) < -1e-6 * max(1.0, abs(means[-1]))):
                bad.append(f"{f} decreases with lead time")
            line += "  (accumulated)"
        print(line)
    print(f"   {'ok' if not bad else 'PROBLEMS: ' + '; '.join(bad)}")
    ok &= not bad

    print(f"\nVERDICT  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
