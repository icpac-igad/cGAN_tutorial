## About this run

How long one day of the SEWAA cGAN inputs takes on one machine, end to end, up
to the point where the trained model runs. Nine 00z cycles, 20261021 to
20261029, were run one after the other on 2026-10-30, as a daily job would run
them. The summary and the per-cycle sections are generated from the run's own
log, timing file and stores by [`make_local_run_report.py`](make_local_run_report.py)
(none of their numbers is typed in); the sections from here to "Notes" come from
[`2026-10-30-local-9day-intro.md`](2026-10-30-local-9day-intro.md).

## What was run

| | |
|---|---|
| Code | `cGAN_tutorial/ifs-0p1-gik` at commit **`57ee2cc`** (step4 `--no-cluster`, per-stage timing), on top of `1ff14c2` (the SEWAA products) |
| Driver | `run_local.sh <YYYYMMDD>` once per date, with `BUILD_VD=1 PRODUCTS="jurre-6h mvua-7d"`, the cycles in sequence (below) |
| Machine | `jupyterhub-gateway2`, AMD EPYC 7713, 8 vCPU, 31 GB RAM; stores and outputs on a local 100 GB disk (`/srv/state`) |
| Cluster | **none**: step4 runs its block tasks on 12 threads inside one process (`--no-cluster --threads 12`, 4 GETs per block); no Dask, Frisky or EWC workers |
| Uploads | none; nothing is written outside this machine |
| Python | 3.12.15: icechunk 2.1.1, zarr 3.2.1, xarray 2026.7.0, gribberish 1.6.0, numpy 2.5.1, imagecodecs 2026.10.10 (libaec), netCDF4 1.7.4 |
| Data | ECMWF open data, IFS ENS 0.1°, `s3://ecmwf-forecasts` with the byte-identical `gs://ecmwf-open-data` copy as fallback; read anonymously |

```bash
export PY=/srv/state/ifs0p1/venv/bin/python TFPY=/srv/state/ifs0p1/venv-infer/bin/python
export IFS0P1_LOCAL_ROOT=/srv/state/ifs0p1/icechunk OUTDIR=/srv/state/ifs0p1/nc9 GAN_OUTDIR=/srv/state/ifs0p1/gan9
export TIMING=/srv/state/ifs0p1/timing9.tsv BUILD_VD=1 PRODUCTS="jurre-6h mvua-7d"
for d in 21 22 23 24 25 26 27 28 29; do bash run_local.sh 202610$d; done
```

## The stages, per cycle

| Stage | Script | What it does |
|---|---|---|
| step1 | `step1_build_virtual.py --verify 8` | reads the cycle's 340 `.index` files from GCS and adds ~742k byte-range references to a **local virtual Icechunk store** (`ecmwf-ifs-0p1-icechunk-vd`); 8 chunks read back and compared with a direct decode |
| step4 | `step4_realize.py --product jurre-6h mvua-7d --no-cluster --threads 12` | **local realized Icechunk store**, one per cycle: 14 fields × the 30 lead times 0..174 h the two products need × 51 members over 25°N–15°S, 19–55°E (401 × 361). Each message is a prefix read (message start → the box's last row, decoded with libaec), ~67% of its bytes |
| step5 | `step5_check.py --samples 8` | every chunk written; 8 random boxes bit-exact against whole messages decoded through the virtual store by gribberish; fields finite and plausible |
| step6 | `step6_sewaa_netcdf.py --product jurre-6h mvua-7d` | the two IFS NetCDFs the SEWAA scripts read: ensemble mean and sd of 51 members (sd ddof 0) on the 384 × 352 cGAN frame; `cgan-ifs-6h-ens/` (+30..54 h) and `cgan-ifs-7d-ens/` (+0..174 h) |
| step7 | `step7_infer.py --dry-run`, per product | builds every network input (jurre-6h: 4 windows × 56 channels; mvua-7d: 7 days × 28 channels) and checks it is finite. **The model is not run** |

## Weights: not used in this run

No trained weights were loaded; step7 ran as `--dry-run`, so the times stop
just before the generator. The weights were located during the run:

- **6 h (Jurre-Brishti):** `Fenwick-Cooper/SEWAA-forecasts` @ `2fe8a40`,
  `6h_accumulations/cGAN/ICPAC-big-ensmeansd/models/gen_weights-0316800.h5`. This is
  the operational checkpoint 316800 (`sewaa-forecasts-package`
  `configs/jurre-brishti/forecast-ens.yaml`), same 14 fields and 56-channel
  input; it loads into `vendor/ensemble_cgan` with 56 input channels. Its
  `FCSTNorm2018.pkl` and `cGAN_data/elev.nc`, `lsm.nc` have the same md5 as the
  constants `run_local.sh` uses.
- **7 days:** the operational Mvua-Kubwa checkpoint 300800 (14 fields, 28
  channels) is not public. SEWAA-forecasts has newer per-lead-day 24 h models
  (`24h_accumulations/cGAN/logs_1` @ 203776 day 0, `logs_5` @ 163840 day 1,
  `logs_17` @ 115200 days 2–6) with **13 fields (no cape), 26 channels** and
  their own normalisation. They load only with 26 input channels, so they need
  a separate step7 mode; the `cgan-ifs-7d-ens` NetCDF already holds what they read.

## Notes

- **AWS throttling.** Most messages came from the GCS copy: the reader falls
  back to it on any S3 error, and S3 answered 503 Slow Down for most GETs on
  this day. step4's time is mostly network time (mean GET ~1.1–1.3 s, decode
  ~35 ms), so it varies with ECMWF's buckets more than with this machine.
- **step1 for every cycle.** The public virtual store on source.coop already
  holds 20261021–26; it was rebuilt locally for every cycle so that each day
  is timed the same way, as a daily job would have to build it.
- The per-stage seconds are also in [`2026-10-30-local-9day-timing.tsv`](2026-10-30-local-9day-timing.tsv).
