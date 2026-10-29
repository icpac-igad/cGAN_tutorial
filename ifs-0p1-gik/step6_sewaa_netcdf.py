"""STEP 6 (SEWAA) -- realized Icechunk store -> the IFS NetCDF the SEWAA
operational cGAN scripts read: ensemble mean + sd, already reduced over members.

One file per product and cycle, in a folder named as the SEWAA jobs name it:

    <outdir>/cgan-ifs-6h-ens/IFS_<YYYYMMDD>_<HH>Z.nc    --product jurre-6h
    <outdir>/cgan-ifs-7d-ens/IFS_<YYYYMMDD>_<HH>Z.nc    --product mvua-7d

      time        (time=1)          hours since 1900-01-01 00:00:00.0
      valid_time  (valid_time)      jurre-6h: init + 30..54 h (5); mvua-7d: init + 0..174 h (30)
      latitude    (384) ascending -13.65..24.65; longitude (352) 19.15..54.25
      <field>_ensemble_mean                float32 (valid_time, latitude, longitude)
      <field>_ensemble_standard_deviation  float32 (valid_time, latitude, longitude)

for the 14 fields cape cp mcc sp ssr t2m tciw tclw tcrw tcw tcwv tp u700 v700,
on the TF cGAN frame (each point the mean of the 4 surrounding 0.1 deg points,
as step6_make_cgan_netcdf.py --frame tf).

What the reference scripts need (vendor/PROVENANCE.md, vendor/ensemble_cgan/reference/):
  - Accumulated fields (tp, cp, ssr) are PRE-DIFFERENCED: index j holds the
    increment valid_time j -> j+1, per member, then mean/sd over members
    (data.py load_fcst). The last index has no j+1 and is NaN; neither script
    reads it (jurre-6h uses only the window start, mvua-7d indices 1..28).
  - jurre-6h: test_forecast.py hard-codes that the file starts at +30 h.
  - mvua-7d: forecast_date.py hard-codes indices 1..29 of a file from +0 h.
Units stay as published (m, J m-2, ...): both scripts convert themselves.

Not stated in any of the source repositories, so set here and recorded in the file:
--members all (control + 50, default) and --ddof 0 (population sd, default).

The fields are read once for all products together: one field at a time, every
lead time any product needs, --threads at once (~55 MB per lead time on the frame,
float64; ~1.7 GB for mvua-7d's 30).

  python step6_sewaa_netcdf.py --store local:$HOME/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261026T00z-sewaa \\
      --product jurre-6h mvua-7d --outdir nc
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cgan_dag  # noqa: F401,E402 -- strips AWS_* before any store opens
import config as C  # noqa: E402
import stores as S  # noqa: E402
from step6_make_cgan_netcdf import TF_LAT, TF_LON, UNITS, tf_frame_index, to_tf_frame  # noqa: E402

import numpy as np  # noqa: E402
import warnings     # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning, module="netCDF4")
warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")


def reduce_product(blocks, steps, field, ddof):
    """{lead h: (member, 384, 352)} -> mean, sd, each (len(steps), 384, 352)."""
    n = len(steps)
    mean = np.full((n,) + blocks[steps[0]].shape[1:], np.nan, np.float32)
    sd = np.full_like(mean, np.nan)
    for k, s in enumerate(steps):
        if field in C.ACCUMULATED:
            if k + 1 == n:
                break                                   # no j+1: stays NaN, never read
            x = blocks[steps[k + 1]] - blocks[s]
        else:
            x = blocks[s]
        mean[k] = x.mean(axis=0)
        sd[k] = x.std(axis=0, ddof=ddof)
    return mean, sd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="realized store spec")
    ap.add_argument("--env", default=None, help="Ceph AK/SK file for a ceph: store")
    ap.add_argument("--product", nargs="+", choices=list(C.PRODUCTS),
                    default=list(C.PRODUCTS))
    ap.add_argument("--outdir", default="nc", help="the product folders go under here")
    ap.add_argument("--members", default="all",
                    help="all (0..50, default), perturbed (1..50), or a count N (first N perturbed)")
    ap.add_argument("--ddof", type=int, default=0, help="sd over members: 0 population, 1 sample")
    ap.add_argument("--complevel", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    import netCDF4 as nc
    ds, snap = S.open_realized_dataset(a.store, a.env)
    have = [int(m) for m in ds.number.values]
    if a.members == "all":
        members = have
    elif a.members == "perturbed":
        members = [m for m in have if m != 0]
    else:
        members = [m for m in have if m != 0][:int(a.members)]
    midx = [have.index(m) for m in members]
    missing = [f for f in C.FIELDS if f not in ds]
    if missing:
        sys.exit(f"store lacks {missing}")
    steps_h = [int(s / np.timedelta64(1, "h")) for s in ds.step.values]
    for p in a.product:
        lack = sorted(set(C.PRODUCTS[p]["steps"]) - set(steps_h))
        if lack:
            sys.exit(f"{p} needs {lack} h, not in the store (realize with "
                     f"step4_realize.py --product {p}; 06z/18z stop at 144 h)")
    need = sorted({s for p in a.product for s in C.PRODUCTS[p]["steps"]})

    init = np.datetime64(ds.time.values[0], "h")
    h1900 = int((init - np.datetime64("1900-01-01T00", "h")) / np.timedelta64(1, "h"))
    stamp = str(init)
    date8, hh = stamp[:10].replace("-", ""), stamp[11:13]
    if "mvua-7d" in a.product and hh != "00":
        print(f"WARNING: forecast_date.py names its output GAN_<date>_00Z.nc whatever "
              f"the input: a {hh}z mvua-7d forecast would be labelled 00Z")
    lat = ds.latitude.values
    flip = lat[0] > lat[-1]
    j0, i0 = tf_frame_index(lat[::-1] if flip else lat, ds.longitude.values)

    t0 = time.time()
    roots, outs = {}, {}
    for p in a.product:
        d = os.path.join(a.outdir, C.PRODUCTS[p]["dir"])
        os.makedirs(d, exist_ok=True)
        outs[p] = os.path.join(d, f"IFS_{date8}_{hh}Z.nc")
        steps = C.PRODUCTS[p]["steps"]
        r = nc.Dataset(outs[p] + ".part", "w", format="NETCDF4")
        r.description = ("ECMWF IFS ENS 0.1 deg ensemble mean and standard deviation "
                         f"over East Africa, SEWAA cGAN input ({p})")
        r.cgan_product = p
        r.members = f"{len(members)}: {members[0]}..{members[-1]}"
        r.sd_ddof = a.ddof
        r.accumulated = ("tp cp ssr: index j holds the per-member increment valid_time j -> "
                         "j+1, then mean/sd over members; the last index is NaN")
        r.source_store = a.store
        r.source_snapshot = snap
        r.virtual_store = str(ds.attrs.get("virtual_store"))
        r.virtual_store_snapshot = str(ds.attrs.get("virtual_store_snapshot"))
        r.frame = ("TF cGAN frame 384x352, -13.65..24.65N 19.15..54.25E: mean of the "
                   "4 surrounding 0.1 deg points")
        r.workflow = "cGAN_tutorial/ifs-0p1-gik step6_sewaa_netcdf.py (vendor/PROVENANCE.md)"
        r.createDimension("time", 1)
        r.createDimension("valid_time", len(steps))
        r.createDimension("latitude", len(TF_LAT))
        r.createDimension("longitude", len(TF_LON))
        v = r.createVariable("time", "f8", ("time",))
        v.units, v[:] = "hours since 1900-01-01 00:00:00.0", [h1900]
        v = r.createVariable("valid_time", "f8", ("valid_time",))
        v.units, v[:] = "hours since 1900-01-01 00:00:00.0", [h1900 + s for s in steps]
        v = r.createVariable("latitude", "f4", ("latitude",))
        v.units, v[:] = "degrees_north", TF_LAT
        v = r.createVariable("longitude", "f4", ("longitude",))
        v.units, v[:] = "degrees_east", TF_LON
        roots[p] = r
    print(f"store    {a.store}  snapshot {snap}")
    print(f"cycle    {date8} {hh}z  {len(members)} members  {len(need)} lead times "
          f"{need[0]}..{need[-1]} h  -> {' '.join(a.product)}")

    try:
        for f in C.FIELDS:
            def fetch(s, f=f):
                # float64 before the 4-point mean: ssr accumulations reach 4e7 J m-2,
                # where float32 spacing (4 J) is the size of a night-time increment
                b = ds[f].isel(time=0, step=steps_h.index(s)).values[midx].astype(np.float64)
                return s, to_tf_frame(b[:, ::-1, :] if flip else b, j0, i0)

            with cf.ThreadPoolExecutor(a.threads) as ex:
                blocks = dict(ex.map(fetch, need))
            for s, b in blocks.items():
                if not np.isfinite(b).all():
                    sys.exit(f"{f} +{s}h has non-finite values -- run step5_check.py")
            for p, r in roots.items():
                mean, sd = reduce_product(blocks, C.PRODUCTS[p]["steps"], f, a.ddof)
                for kind, arr in (("mean", mean), ("standard_deviation", sd)):
                    v = r.createVariable(
                        f"{f}_ensemble_{kind}", "f4", ("valid_time", "latitude", "longitude"),
                        zlib=a.complevel > 0, complevel=a.complevel or None,
                        shuffle=a.complevel > 0, chunksizes=(1, len(TF_LAT), len(TF_LON)))
                    v.units = UNITS.get(f, "")
                    if f in C.ACCUMULATED:
                        v.comment = "6 h increment from this valid time to the next"
                    v[:] = arr
            print(f"  {f:5s} {len(need)} lead times x {len(members)} members  "
                  f"{time.time() - t0:6.0f}s", flush=True)
    except BaseException:
        for p, r in roots.items():
            r.close()
            os.remove(outs[p] + ".part")
        raise
    for p, r in roots.items():
        r.close()
        os.replace(outs[p] + ".part", outs[p])
        print(f"wrote {outs[p]}  {os.path.getsize(outs[p]) / 1e6:.0f} MB")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
