"""STEP 6 -- realized Icechunk store -> the cGAN inference NetCDF.

Writes ONE file per cycle, in the layout scripts/ensemble-forecast.py reads:

    IFS_<YYYYMMDD>_<HH>Z.nc
      time        (time=1)          hours since 1900-01-01 00:00:00.0
      valid_time  (valid_time=29)   hours since 1900-01-01, init + 0..168 h step 6
      member      (member)          ensemble member number
      latitude    (latitude)        ASCENDING (S -> N), as the training files
      longitude   (longitude)
      <field>_ensemble_mean   float32 (time, member, valid_time, latitude, longitude)

for the 14 fields cape cp mcc sp ssr t2m tciw tclw tcrw tcw tcwv tp u700 v700.
Despite the name, `<field>_ensemble_mean` holds every MEMBER: that is the name
ensemble-forecast.py reads (`nc_in[f"{field}_ensemble_mean"][0, :, i:i+2]`),
and it takes the mean/sd over the member axis itself. Values are as published:
tp/cp in m and ssr in J m-2 ACCUMULATED from the init -- the inference
differences consecutive valid times itself (accumulated_fields in data.py).

Reads one field at a time (29 chunks, fetched --threads at once), so memory
stays at ~29 x 27 MB = 0.8 GB whatever the store's size. The file is large: 14 x 50 x 29 x 401 x 331 float32
is ~10.8 GB raw, ~3-5 GB with the zlib level 1 used here. --members and
--fields cut it down for tests.

  python step6_make_cgan_netcdf.py --store local:$HOME/icechunk-local/ecmwf-ifs-0p1-cgan14-20e53e-25n15s/2026102100z --outdir nc
  python step6_make_cgan_netcdf.py --store sourcecoop:forecasts/ecmwf-ifs-0p1-cgan14-20e53e-25n15s/2026102100z --outdir nc
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

import numpy as np  # noqa: E402
import warnings     # noqa: E402

# netCDF4 1.7.4 sets ndarray.shape internally, deprecated in NumPy 2.5: harmless
warnings.filterwarnings("ignore", category=DeprecationWarning, module="netCDF4")
warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")

UNITS = {"cape": "J kg**-1", "cp": "m", "mcc": "(0 - 1)", "sp": "Pa", "ssr": "J m**-2",
         "t2m": "K", "tciw": "kg m**-2", "tclw": "kg m**-2", "tcrw": "kg m**-2",
         "tcw": "kg m**-2", "tcwv": "kg m**-2", "tp": "m", "u700": "m s**-1",
         "v700": "m s**-1"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="realized store spec")
    ap.add_argument("--env", default=None, help="Ceph AK/SK file for a ceph: store")
    ap.add_argument("--outdir", default="nc")
    ap.add_argument("--members", default="perturbed",
                    help="perturbed (1..50, the IFS ENS files the model was run on), "
                         "all (0..50), or a count N (first N perturbed)")
    ap.add_argument("--fields", nargs="+", default=list(C.FIELDS))
    ap.add_argument("--complevel", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6,
                    help="chunks read ahead concurrently (each ~27 MB in memory)")
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
    missing = [f for f in a.fields if f not in ds]
    if missing:
        sys.exit(f"store lacks {missing}")
    steps_h = [int(s / np.timedelta64(1, "h")) for s in ds.step.values]
    want = [s for s in C.STEPS if s in steps_h]
    if want != C.STEPS:
        print(f"WARNING: store has {len(want)} of the {len(C.STEPS)} cGAN lead times")
    sidx = [steps_h.index(s) for s in want]
    midx = [have.index(m) for m in members]

    init = np.datetime64(ds.time.values[0], "h")
    h1900 = int((init - np.datetime64("1900-01-01T00", "h")) / np.timedelta64(1, "h"))
    stamp = str(init.astype("datetime64[h]"))
    date8, hh = stamp[:10].replace("-", ""), stamp[11:13]
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"IFS_{date8}_{hh}Z.nc")
    tmp = out + ".part"
    lat = ds.latitude.values
    flip = lat[0] > lat[-1]
    lat_out = lat[::-1] if flip else lat

    t0 = time.time()
    root = nc.Dataset(tmp, "w", format="NETCDF4")
    root.description = ("ECMWF IFS ENS 0.1 deg over East Africa for cGAN inference "
                        "(ensemble-forecast.py layout)")
    root.source_store = a.store
    root.source_snapshot = snap
    root.virtual_store = str(ds.attrs.get("virtual_store"))
    root.virtual_store_snapshot = str(ds.attrs.get("virtual_store_snapshot"))
    root.workflow = "cGAN_tutorial/ifs-0p1-gik"
    root.createDimension("time", 1)
    root.createDimension("member", len(members))
    root.createDimension("valid_time", len(want))
    root.createDimension("latitude", len(lat_out))
    root.createDimension("longitude", ds.sizes["longitude"])
    v = root.createVariable("time", "f8", ("time",))
    v.units, v[:] = "hours since 1900-01-01 00:00:00.0", [h1900]
    v = root.createVariable("valid_time", "f8", ("valid_time",))
    v.units, v[:] = "hours since 1900-01-01 00:00:00.0", [h1900 + s for s in want]
    v = root.createVariable("member", "i4", ("member",))
    v[:] = members
    v = root.createVariable("latitude", "f4", ("latitude",))
    v.units, v[:] = "degrees_north", lat_out
    v = root.createVariable("longitude", "f4", ("longitude",))
    v.units, v[:] = "degrees_east", ds.longitude.values
    for f in a.fields:
        var = root.createVariable(
            f"{f}_ensemble_mean", "f4", ("time", "member", "valid_time", "latitude",
                                         "longitude"),
            zlib=a.complevel > 0, complevel=a.complevel or None,
            chunksizes=(1, 1, 1, len(lat_out), ds.sizes["longitude"]))
        var.units = UNITS.get(f, "")
        var.long_name = f"{f}, every ensemble member (name as read by ensemble-forecast.py)"
        if f in C.ACCUMULATED:
            var.comment = "accumulated from the forecast init; difference consecutive valid times"
        def fetch(si, f=f):
            return ds[f].isel(time=0, step=si).values[midx]       # (member, lat, lon)

        with cf.ThreadPoolExecutor(a.threads) as ex:              # read ahead, write in order
            blocks = list(ex.map(fetch, sidx))
        for k, block in enumerate(blocks):
            if flip:
                block = block[:, ::-1, :]
            if not np.isfinite(block).all():
                root.close()
                os.remove(tmp)
                sys.exit(f"{f} +{want[k]}h has non-finite values -- run step5_check.py")
            var[0, :, k, :, :] = block
        print(f"  {f:5s} {len(want)} valid times x {len(members)} members  "
              f"{time.time() - t0:6.0f}s", flush=True)
    root.close()
    os.replace(tmp, out)
    print(f"wrote {out}  {os.path.getsize(out) / 1e9:.2f} GB  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
