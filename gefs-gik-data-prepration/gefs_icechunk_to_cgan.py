# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1", "gribberish>=1.4",
#   "numpy", "h5netcdf", "h5py",
# ]
# ///
"""cGAN forecast inputs straight out of the GEFS Icechunk store.

Replaces stages 1-4 of run_gefs_gik_cgan_pipeline.py with one read:

    1  download the GIK template from HuggingFace   |
    2  build parquet references for the date        |  already materialised
    3  stream the variables from S3 into zarr       |  in the Icechunk store
    4  convert zarr -> per-field NetCDF             |

The store IS the product of 1-3: a virtual Zarr over the same
`s3://noaa-gefs-pds` GRIB bytes with every (date, member, step, variable)
already referenced. Nothing is re-derived here -- this selects and writes.
Measured: 30 s for a full 30-member init against the README's ~2 h for 1-3.

Output is what run_gefs_inference_raw.py already reads, so stage 5 is unchanged:

    {field}_{YYYYMMDD}_{RUN}z.nc   dims (time, member, step, latitude, longitude)

Latitude stays in the store's native N->S order and is labelled honestly; the
inference flips to ascending itself when `lat[0] > lat[-1]`.

APCP is the per-bucket value at the 6-hourly boundary steps, NOT cumulative --
see APCP_BUCKET_VS_CUMULATIVE.md: the model was trained per-bucket and
cumulative inflates the input ~5-9x. The steps selected here (30/36/42/48/54 h)
ARE bucket boundaries, so the stored value is already the full 6 h window.

Executors
---------
    threads  one process, a thread pool over (member, step)  -- one init
    frisky   a Frisky cluster, one task per (date, field)    -- many inits

Usage:
  uv run gefs_icechunk_to_cgan.py --date 20250918 --outdir netcdf/2025
  uv run gefs_icechunk_to_cgan.py --dates 20250915:20250918 --executor frisky
  uv run gefs_icechunk_to_cgan.py --date 20250918 --store gcs --sa-key key.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
from datetime import datetime, timedelta, timezone

# The AWS_* trap: icechunk builds the virtual-chunk S3 client lazily and caches
# object-store config process-wide, so AWS_* must be gone BEFORE anything opens
# the store -- here and in every worker. Frisky's LocalCluster spawns workers
# from this process's environment, so popping here covers both.
_STRIPPED_AWS = {k: os.environ.pop(k) for k in list(os.environ)
                 if k.startswith("AWS_")}

import numpy as np                                            # noqa: E402
import xarray as xr                                           # noqa: E402
import icechunk                                               # noqa: E402
import zarr                                                   # noqa: E402
import gribberish.zarr  # noqa: F401,E402 -- registers the Zarr v3 codec

GROUP = "0p25/00z"
CONTAINER = "s3://noaa-gefs-pds/"

# cGAN field -> Icechunk array. The same physical messages the parquet path
# used (CGAN_VARIABLES in stream_gefs_for_cgan.py), by store array name.
FIELDS = {
    "cape": "cape",    # CAPE:surface
    "pres": "sp",      # PRES:surface
    "pwat": "pwat",    # PWAT:entire atmosphere
    "tmp":  "t2m",     # TMP:2 m above ground
    "ugrd": "u10",     # UGRD:10 m above ground
    "vgrd": "v10",     # VGRD:10 m above ground
    "msl":  "mslet",   # MSLET:mean sea level
    "apcp": "tp",      # APCP:surface
}
FIELD_ORDER = ["cape", "pres", "pwat", "tmp", "ugrd", "vgrd", "msl", "apcp"]

# ICPAC / cGAN domain, as in run_gefs_inference_raw.py. Both longitudes are
# positive east, so they need no wrapping on the store's 0-360 axis (GEFS starts
# at 0 deg E, unlike ECMWF -- see grib-index-kerchunk/gefs/grids.py).
LAT_MIN, LAT_MAX = -13.65, 24.65
LON_MIN, LON_MAX = 19.15, 54.25

VALID_TIMES = [30, 36, 42, 48]   # cGAN output steps
HOURS = 6                        # training pairs (t, t+HOURS)
STEP_INTERVAL = 3                # GEFS step spacing

PUBLISHED = dict(bucket="e4drr-project",
                 prefix="forecasts/noaa_gefs_aws_s3_icechunk_vd",
                 endpoint="https://data.source.coop")
GCS_STORE = ("gik-gefs-aws-tf", "icechunk/gefs-ens")


def step_hours() -> list[int]:
    """The forecast hours inference asks for: each valid time and +HOURS."""
    return sorted({h for v in VALID_TIMES for h in (v, v + HOURS)})


_GROUP_CACHE: dict = {}


def open_group(store_kind: str, sa_key: str | None, attempts: int = 10):
    """Opened once per process and reused.

    Opening is not cheap -- on source.coop the manifest preload is disabled to
    survive its sporadic 500s, so a cold open costs ~60 s. Re-opening per
    (date, field) meant 8 opens per init and dominated the wall clock. The cache
    is per process, so Frisky workers amortise it across their whole task run.
    """
    ck = (store_kind, sa_key)
    if ck in _GROUP_CACHE:
        return _GROUP_CACHE[ck]
    auth = icechunk.containers_credentials(
        {CONTAINER: icechunk.s3_anonymous_credentials()})
    if store_kind == "gcs":
        storage = icechunk.gcs_storage(bucket=GCS_STORE[0], prefix=GCS_STORE[1],
                                       service_account_file=sa_key)
        repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=auth)
    else:
        storage = icechunk.s3_storage(
            bucket=PUBLISHED["bucket"], prefix=PUBLISHED["prefix"],
            endpoint_url=PUBLISHED["endpoint"], region="us-east-1",
            anonymous=True, from_env=False, force_path_style=True)
        cfg = icechunk.RepositoryConfig.default()
        cfg.manifest = icechunk.ManifestConfig(
            preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                                   max_arrays_to_scan=0))
        repo = last = None
        for i in range(attempts):            # source.coop throws sporadic 500s
            try:
                repo = icechunk.Repository.open(
                    storage, config=cfg, authorize_virtual_chunk_access=auth)
                break
            except Exception as e:           # noqa: BLE001
                last = e
                time.sleep(2 + 2 * i)
        if repo is None:
            raise SystemExit(f"cannot open source.coop store: {last}")
    g = zarr.open_group(store=repo.readonly_session("main").store, path=GROUP,
                        mode="r", zarr_format=3)
    _GROUP_CACHE[ck] = g
    return g


_LOCATE_CACHE: dict = {}


def locate(g, date: str):
    """(time index, row slice, col slice, region lat, region lon), cached."""
    lk = (id(g), date)
    if lk in _LOCATE_CACHE:
        return _LOCATE_CACHE[lk]
    tvals = np.asarray(g["time"][:])
    tstr = np.array([datetime.fromtimestamp(int(t) * 3600, tz=timezone.utc)
                     .strftime("%Y%m%d") for t in tvals])
    hit = np.where(tstr == date)[0]
    if not hit.size:
        raise SystemExit(f"{date} is not in the store ({tstr.min()}..{tstr.max()})")
    lat, lon = np.asarray(g["latitude"][:]), np.asarray(g["longitude"][:])
    li = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    lj = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
    la, lb = int(li[0]), int(li[-1]) + 1
    oa, ob = int(lj[0]), int(lj[-1]) + 1
    out = (int(hit[0]), (la, lb), (oa, ob), lat[la:lb], lon[oa:ob])
    _LOCATE_CACHE[lk] = out
    return out


def extract_field(date: str, field: str, outdir: str, run: str = "00",
                  members: int = 30, store_kind: str = "source-coop",
                  sa_key: str | None = None, threads: int = 16) -> dict:
    """Write one {field}_{date}_{run}z.nc. Self-contained: safe on a worker."""
    g = open_group(store_kind, sa_key)
    ti, (la, lb), (oa, ob), rlat, rlon = locate(g, date)
    hours = step_hours()
    sidx = [h // STEP_INTERVAL for h in hours]
    arr = g[FIELDS[field]]

    out = np.full((members, len(sidx), rlat.size, rlon.size), np.nan, "float32")

    def grab(job):
        m, k, s = job
        return m, k, arr[ti, m, s, la:lb, oa:ob]

    jobs = [(m, k, s) for m in range(members) for k, s in enumerate(sidx)]
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        for m, k, block in ex.map(grab, jobs):
            out[m, k] = block

    ds = xr.Dataset(
        {field: (("time", "member", "step", "latitude", "longitude"),
                 out[None, ...])},
        coords={"time": [np.datetime64(
                    datetime.strptime(date, "%Y%m%d").date(), "ns")],
                "member": np.arange(1, members + 1, dtype="int32"),
                "step": np.array(hours, dtype="int32"),
                "latitude": rlat.astype("float32"),
                "longitude": rlon.astype("float32")},
        attrs={"source": f"GEFS Icechunk store ({store_kind}) group {GROUP}",
               "grib": FIELDS[field],
               "note": "APCP is per-bucket at 6-hourly boundaries, not cumulative"})
    ds["step"].attrs["units"] = "hours"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{field}_{date}_{run}z.nc")
    ds.to_netcdf(path, engine="h5netcdf")
    return dict(path=path, field=field, date=date, shape=list(out.shape),
                finite=float(np.isfinite(out).mean() * 100),
                mean=float(np.nanmean(out)))


def parse_dates(a) -> list[str]:
    if a.date:
        return [a.date]
    if ":" in a.dates:
        s, e = a.dates.split(":")
        d0 = datetime.strptime(s, "%Y%m%d").date()
        d1 = datetime.strptime(e, "%Y%m%d").date()
        n = (d1 - d0).days
        return [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range(n + 1)]
    return [d.strip() for d in a.dates.split(",") if d.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    gsel = ap.add_mutually_exclusive_group(required=True)
    gsel.add_argument("--date", help="single init date YYYYMMDD")
    gsel.add_argument("--dates", help="comma list, or START:END inclusive")
    ap.add_argument("--run", default="00")
    ap.add_argument("--outdir", default="netcdf")
    ap.add_argument("--members", type=int, default=30)
    ap.add_argument("--store", choices=["source-coop", "gcs"], default="source-coop")
    ap.add_argument("--sa-key", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--executor", choices=["threads", "frisky"], default="threads")
    ap.add_argument("--workers", type=int, default=4, help="frisky LocalCluster size")
    ap.add_argument("--scheduler", default=None,
                    help="existing Frisky scheduler address; default spawns local")
    a = ap.parse_args()

    dates = parse_dates(a)
    hours = step_hours()
    print(f"cGAN valid times {VALID_TIMES} (+{HOURS}h) -> hours {hours} -> "
          f"step idx {[h // STEP_INTERVAL for h in hours]}")
    print(f"{len(dates)} init(s): {dates[0]}..{dates[-1]}  "
          f"{a.members} members  executor={a.executor}")
    if _STRIPPED_AWS:
        print(f"  (stripped {len(_STRIPPED_AWS)} AWS_* vars before opening store)")

    t0 = time.time()
    jobs = [(d, f) for d in dates for f in FIELD_ORDER]
    results = []

    if a.executor == "threads":
        for d, f in jobs:
            r = extract_field(d, f, a.outdir, a.run, a.members, a.store,
                              a.sa_key, a.threads)
            results.append(r)
            print(f"  {r['date']} {r['field']:<5} {tuple(r['shape'])} "
                  f"finite {r['finite']:5.1f}%  mean {r['mean']:>12.3f}")
    else:
        import frisky
        if a.scheduler:
            client = frisky.Client(a.scheduler)
            print(f"  frisky: attached to {a.scheduler}")
        else:
            cluster = frisky.LocalCluster(n_workers=a.workers)
            client = cluster.get_client()   # NOT frisky.Client(cluster)
            print(f"  frisky: local cluster, {a.workers} workers")
        futs = [client.submit(extract_field, d, f, a.outdir, a.run, a.members,
                              a.store, a.sa_key, a.threads) for d, f in jobs]
        for fut in futs:
            r = fut.result()
            results.append(r)
            print(f"  {r['date']} {r['field']:<5} {tuple(r['shape'])} "
                  f"finite {r['finite']:5.1f}%  mean {r['mean']:>12.3f}")

    el = time.time() - t0
    nchunks = len(dates) * a.members * len(hours) * len(FIELD_ORDER)
    print(f"\n{len(results)} files for {len(dates)} init(s) in {el:.0f}s "
          f"({el / max(len(dates), 1):.0f}s per init, {nchunks:,} chunks)")
    print(f"outdir: {os.path.abspath(a.outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
