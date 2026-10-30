# Local end-to-end run, 9 daily cycles 20261021–29 00z, no cluster

## Summary, seconds per stage

| Cycle | step1 virtual store | step4 realize | step5 check | step6 NetCDF | step7 jurre-6h | step7 mvua-7d | **total** | step4 msg/s | fetched GB | step5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20261021 00z | 49 | 747 | 19 | 37 | 2 | 2 | **856** (14.3 min) | 28.7 | 29.39 | PASS |
| 20261022 00z | 48 | 637 | 21 | 37 | 1 | 3 | **747** (12.4 min) | 33.7 | 29.34 | PASS |
| 20261023 00z | 48 | 568 | 20 | 39 | 2 | 3 | **680** (11.3 min) | 37.7 | 29.34 | PASS |
| 20261024 00z | 48 | 593 | 21 | 39 | 2 | 3 | **706** (11.8 min) | 36.2 | 29.14 | PASS |
| 20261025 00z | 47 | 551 | 20 | 39 | 3 | 2 | **662** (11.0 min) | 39.0 | 29.53 | PASS |
| 20261026 00z | 47 | 561 | 20 | 41 | 2 | 3 | **674** (11.2 min) | 38.2 | 29.74 | PASS |
| 20261027 00z | 48 | 563 | 21 | 48 | 3 | 5 | **688** (11.5 min) | 38.1 | 29.90 | PASS |
| 20261028 00z | 49 | 558 | 21 | 44 | 1 | 7 | **680** (11.3 min) | 38.4 | 30.10 | PASS |
| 20261029 00z | 49 | 576 | 21 | 47 | 2 | 8 | **703** (11.7 min) | 37.3 | 30.17 | PASS |

9 cycles: total mean **11.8 min**, min 11.0, max 14.3; all 1.78 h.

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
  back to it on any S3 error, and S3 answered 503 Slow Down for most GETs
  throughout the run. step4's time is mostly network time (mean GET ~1.1–1.3 s, decode
  ~35 ms), so it varies with ECMWF's buckets more than with this machine.
- **step1 for every cycle.** The public virtual store on source.coop already
  holds 20261021–26; it was rebuilt locally for every cycle so that each day
  is timed the same way, as a daily job would have to build it.
- **On disk after the 9 cycles:** realized stores 26 GB (~2.65 GB per cycle),
  local virtual store 151 MB, the 18 SEWAA NetCDFs 2.8 GB (6 h ~45 MB, 7 d
  ~280 MB per cycle).
- **Log buffering.** During this run `run_local.sh` piped each step through
  `grep` without `--line-buffered`, so step4's progress lines reached the log
  only when the step ended. The times are not affected. Fixed after the run.
- The per-stage seconds are also in [`2026-10-30-local-9day-timing.tsv`](2026-10-30-local-9day-timing.tsv).

## 20261021 00z

Run 2026-10-30T08:28:28Z → 2026-10-30T08:42:44Z, **856 s (14.3 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 49 | 742478 refs from 340 .index files on GCS in 26.6s; virtual snapshot `7FPN0R8A85C4TRR6XTTG`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 747 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 28.7 msg/s; fetched 29.39 of 43.53 GB whole-message bytes (saved 32.5%); 21,404 GETs (4 top-ups), 19,545 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `2FT5SS19SJG1CFNGRYW0`, 2.65 GB on disk |
| step5_check.py `--samples 8` | 19 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 37 | `cgan-ifs-6h-ens/IFS_20261021_00Z.nc` 44 MB, `cgan-ifs-7d-ens/IFS_20261021_00Z.nc` 280 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 2 | dry run  OK: 7 windows of finite 1x384x352x28 input in 2s |

## 20261022 00z

Run 2026-10-30T08:42:44Z → 2026-10-30T08:55:11Z, **747 s (12.4 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 48 | 742478 refs from 340 .index files on GCS in 24.3s; virtual snapshot `GMHHBD3R935X3V1S7KMG`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 637 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 33.7 msg/s; fetched 29.34 of 43.52 GB whole-message bytes (saved 32.6%); 21,400 GETs (0 top-ups), 19,121 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `Q8NPJHPBKK6RGH8XPFS0`, 2.67 GB on disk |
| step5_check.py `--samples 8` | 21 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 37 | `cgan-ifs-6h-ens/IFS_20261022_00Z.nc` 45 MB, `cgan-ifs-7d-ens/IFS_20261022_00Z.nc` 281 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 1 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 3 | dry run  OK: 7 windows of finite 1x384x352x28 input in 2s |

## 20261023 00z

Run 2026-10-30T08:55:11Z → 2026-10-30T09:06:31Z, **680 s (11.3 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 48 | 742478 refs from 340 .index files on GCS in 25.4s; virtual snapshot `N74TDERSEJB8QNK5H9K0`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 568 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 37.7 msg/s; fetched 29.34 of 43.52 GB whole-message bytes (saved 32.6%); 21,401 GETs (1 top-ups), 19,060 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `CD3V269VA7F14C5SJGC0`, 2.67 GB on disk |
| step5_check.py `--samples 8` | 20 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 39 | `cgan-ifs-6h-ens/IFS_20261023_00Z.nc` 46 MB, `cgan-ifs-7d-ens/IFS_20261023_00Z.nc` 282 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 3 | dry run  OK: 7 windows of finite 1x384x352x28 input in 2s |

## 20261024 00z

Run 2026-10-30T09:06:31Z → 2026-10-30T09:18:17Z, **706 s (11.8 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 48 | 742478 refs from 340 .index files on GCS in 24.4s; virtual snapshot `7EAG61CTGY26NRN2H5DG`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 593 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 36.2 msg/s; fetched 29.14 of 43.16 GB whole-message bytes (saved 32.5%); 21,405 GETs (5 top-ups), 18,956 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `Z5EDGE36TDNT8RNXHASG`, 2.67 GB on disk |
| step5_check.py `--samples 8` | 21 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 39 | `cgan-ifs-6h-ens/IFS_20261024_00Z.nc` 46 MB, `cgan-ifs-7d-ens/IFS_20261024_00Z.nc` 281 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 3 | dry run  OK: 7 windows of finite 1x384x352x28 input in 2s |

## 20261025 00z

Run 2026-10-30T09:18:17Z → 2026-10-30T09:29:19Z, **662 s (11.0 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 47 | 742478 refs from 340 .index files on GCS in 25.1s; virtual snapshot `BWTVR9MSQMS84AC8DSRG`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 551 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 39.0 msg/s; fetched 29.53 of 43.52 GB whole-message bytes (saved 32.2%); 21,400 GETs (0 top-ups), 18,467 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `W5VRCDXZJQSFVDWTV3XG`, 2.68 GB on disk |
| step5_check.py `--samples 8` | 20 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 39 | `cgan-ifs-6h-ens/IFS_20261025_00Z.nc` 46 MB, `cgan-ifs-7d-ens/IFS_20261025_00Z.nc` 279 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 3 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 2 | dry run  OK: 7 windows of finite 1x384x352x28 input in 2s |

## 20261026 00z

Run 2026-10-30T09:29:20Z → 2026-10-30T09:40:34Z, **674 s (11.2 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 47 | 742478 refs from 340 .index files on GCS in 24.5s; virtual snapshot `WVTNHK12G3VA56F4ME60`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 561 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 38.2 msg/s; fetched 29.74 of 43.73 GB whole-message bytes (saved 32.0%); 21,404 GETs (4 top-ups), 19,012 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `VEN1ADC2CFD6G8SJB4Y0`, 5.36 GB on disk |
| step5_check.py `--samples 8` | 20 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 41 | `cgan-ifs-6h-ens/IFS_20261026_00Z.nc` 45 MB, `cgan-ifs-7d-ens/IFS_20261026_00Z.nc` 278 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 3 | dry run  OK: 7 windows of finite 1x384x352x28 input in 2s |

## 20261027 00z

Run 2026-10-30T09:40:34Z → 2026-10-30T09:52:02Z, **688 s (11.5 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 48 | 742478 refs from 340 .index files on GCS in 25.5s; virtual snapshot `6BTWSG8Q6Y9VB1TNMP4G`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 563 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 38.1 msg/s; fetched 29.90 of 43.92 GB whole-message bytes (saved 31.9%); 21,400 GETs (0 top-ups), 18,491 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `514HKBVS8ATA6VEFHMYG`, 2.66 GB on disk |
| step5_check.py `--samples 8` | 21 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 48 | `cgan-ifs-6h-ens/IFS_20261027_00Z.nc` 45 MB, `cgan-ifs-7d-ens/IFS_20261027_00Z.nc` 276 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 3 | dry run  OK: 4 windows of finite 1x384x352x56 input in 2s |
| step7_infer.py mvua-7d `--dry-run` | 5 | dry run  OK: 7 windows of finite 1x384x352x28 input in 5s |

## 20261028 00z

Run 2026-10-30T09:52:02Z → 2026-10-30T10:03:22Z, **680 s (11.3 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 49 | 742478 refs from 340 .index files on GCS in 24.3s; virtual snapshot `46Q5NME53X3ZRHYYF990`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 558 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 38.4 msg/s; fetched 30.10 of 44.13 GB whole-message bytes (saved 31.8%); 21,400 GETs (0 top-ups), 18,165 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `K3MMDMQ8PD4RXN7WVK1G`, 2.63 GB on disk |
| step5_check.py `--samples 8` | 21 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 44 | `cgan-ifs-6h-ens/IFS_20261028_00Z.nc` 45 MB, `cgan-ifs-7d-ens/IFS_20261028_00Z.nc` 274 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 1 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 7 | dry run  OK: 7 windows of finite 1x384x352x28 input in 6s |

## 20261029 00z

Run 2026-10-30T10:03:22Z → 2026-10-30T10:15:05Z, **703 s (11.7 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 49 | 742478 refs from 340 .index files on GCS in 24.6s; virtual snapshot `8MBMQ9AJJA7FSFN46Q50`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 576 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 37.3 msg/s; fetched 30.17 of 44.35 GB whole-message bytes (saved 32.0%); 21,412 GETs (12 top-ups), 18,000 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `5N708BM66SPWMMV9HSEG`, 2.60 GB on disk |
| step5_check.py `--samples 8` | 21 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 47 | `cgan-ifs-6h-ens/IFS_20261029_00Z.nc` 45 MB, `cgan-ifs-7d-ens/IFS_20261029_00Z.nc` 273 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 8 | dry run  OK: 7 windows of finite 1x384x352x28 input in 7s |

