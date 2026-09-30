# CHIRPS cGAN — Truth Swap → GPU Training Runbook

Companion to **[RFE2_cGAN_RUNBOOK.md](RFE2_cGAN_RUNBOOK.md)**. Same model, same IFS input,
same training loop — only the truth changes (CHIRPS instead of RFE2). Work folder with the
scripts and full narrative: `/home/ezra/CHIRPS/cGAN_CHIRPS/README.md` — that README is the
source of truth for *why*; this file is the ordered *how*, including what's already done.

> **Golden rule (same as RFE2):** the GPU only ever reads TFRecords + constants from local
> disk. All data prep (truth download, FCSTNorm, TFRecords) is CPU work and must finish before
> the GPU is touched.

---

## 0. Status (as of 2026-07-29)

### Headline: run02 wins

**run02 (`cape` → CHIRPS climatology mean/sd) is the champion.** Best
**CRPS 0.046455 @ checkpoint 181248**. On the 25 eval dates at 10 members it beats
run01 by 2.15% and climatology by 8.4%, and it wins **7 of 7 metrics against each**:

| Metric | **run02** | run01 (`cape`) | Climatology (25d, 10-mem) |
|--------|-----------|----------------|---------------------------|
| **CRPS** | **0.046455** | 0.047475 | 0.050696 |
| CRPS_max_4 | **0.074685** | 0.076852 | 0.081905 |
| CRPS_max_16 | **0.144587** | 0.163211 | 0.151110 |
| CRPS_avg_4 | **0.043642** | 0.044733 | 0.048478 |
| CRPS_avg_16 | **0.038955** | 0.039956 | 0.044354 |
| EMRMSE | **0.174060** | 0.177321 | 0.186088 |
| MAE | **0.071440** | 0.073596 | 0.079038 |

The biggest gain is `max_16` (+11.4% over run01): run01 **lost** to climatology on pooled
extremes, run02 turns that into a win. run02 also converges faster — it reached run01's
*final* score by checkpoint ~67 of 200. Since the RFE2 sweep put field set at ~14% CRPS
versus ~3% for architecture, this is the axis that pays; **the field set is where to keep
looking, not the architecture.**

⚠️ **Both runs BEAT climatology. An earlier note in this file said run01 lost — that was
wrong**, an artifact of scoring the model on 25 dates and the baseline on 364. See §6.

**Shared across all runs** (truth side — never rebuild):

| Stage | State |
|-------|-------|
| CHIRPS truth, 2018–2021 (1461 days) | ✅ DONE — `/home/ezra/CHIRPS_truth/`, audited clean |
| Clone + patch `dsrnngan` | ✅ DONE — `/home/ezra/CHIRPS_cGAN_run/dsrnngan` |
| Rain bins recalibrated for CHIRPS | ✅ DONE — `[0.0082, 0.0439, 0.1234]` mm/hr — truth-derived, so **valid for every run** |
| Climatology baseline, full year | ✅ DONE — CRPS 0.046074, but **8-member/364-date: do NOT compare to a model number as-is** (§6) |
| Climatology baseline, 25 eval dates @ 10 members | ✅ DONE — CRPS **0.050696** — this is the correct yardstick for the 25-date evals |

**run01** — 14 fields incl. `cape`, the RFE2-champion set. COMPLETE:

| Stage | State |
|-------|-------|
| FCSTNorm2018.pkl (14 fields, `cape`) | ✅ DONE — later overwritten by run02; backup at `FCSTNorm2018_run01_14field_cape.bak` |
| TFRecords | ✅ DONE, then **DELETED 2026-07-27** to free 19 GB (rebuildable, see §8) |
| Training | ✅ DONE — 200 checkpoints, finished 2026-07-24 03:18 |
| Evaluation | ✅ DONE — best **CRPS 0.047475 @ ckpt 202752** |
| **Verdict** | ✅ **beats climatology by 6.4%** (0.047475 vs 0.050696), but loses to run02 |

**run02** — `cape` replaced by CHIRPS climatology mean/sd. COMPLETE:

| Stage | State |
|-------|-------|
| `climatology.nc`, 2018–2021 | ✅ DONE — leave-one-out, `train_years=[2018,2019,2020]`, ±7d |
| `data.py` field set (`cape` → `climatology`) | ✅ DONE — still 14 fields × 2 ch = 28 |
| `config_run02.yaml` | ✅ DONE — `log_folder: ../logs_CHIRPS_run02` |
| FCSTNorm2018.pkl (14 fields, `climatology`) | ✅ DONE — rebuilt 2026-07-24 09:54 |
| `data_paths.yaml` → `chirps_tfrecords/run02/` | ✅ DONE |
| TFRecords | ✅ DONE — 16 files, 19 GB, `/home/ezra/chirps_tfrecords/run02/` |
| Training | ✅ DONE — 200 checkpoints, 204,800 samples, 2026-07-29 06:30, ~25.3 h @ 431 s/ckpt |
| Evaluation (25 dates) | ✅ DONE — best **CRPS 0.046455 @ ckpt 181248** |
| Full-year eval (364 dates) | ✅ DONE 2026-07-29 — **CRPS 0.041912**, `logs_CHIRPS_run02/eval_fullyear.txt` |
| **Verdict** | ✅ **CHAMPION** — beats climatology by **6.8% over a full year**, 7/7 metrics |

### Full-year result (the defensible headline)

364 dates of 2021, 10 members on both sides — the comparison to quote:

| Metric | **run02 @ 181248** | Climatology | Margin |
|--------|--------------------|-------------|--------|
| **CRPS** | **0.041912** | 0.044954 | **6.8%** |
| CRPS_max_4 | **0.064277** | 0.069778 | 7.9% |
| CRPS_max_16 | **0.119275** | 0.125936 | 5.3% |
| CRPS_avg_4 | **0.039410** | 0.042875 | 8.1% |
| CRPS_avg_16 | **0.035144** | 0.038851 | 9.5% |
| EMRMSE | **0.158467** | 0.168975 | 6.2% |
| MAE | **0.064191** | 0.071241 | 9.9% |

Files: `logs_CHIRPS_run02/eval_fullyear.txt` and
`CHIRPS/cGAN_CHIRPS/climatology_baseline_2021_10member.txt`.

The full year **validated** the 25-date result rather than overturning it — both systems score
better over the year (25-date: model 0.046455 vs clim 0.050696), because those 25 dates are
harder than average. The margin narrows 8.4% → 6.8%, so the small sample was mildly optimistic,
but the sign and magnitude held. That is what a properly *paired* 25-date comparison should do.

⚠️ **Quote this as a validation number, not a test number** — ckpt 181248 was selected by its
2021 score. See §9.

Don't redo any ✅ step. Nothing is outstanding; §9 records the remaining test-year limitation.

---

## 1. Layout & paths

`dsrnngan/data_paths.yaml`, block `GPU_SERVER` (selected via `data_paths.yaml`'s active key):

```yaml
GPU_SERVER:
    GENERAL:
        TRUTH_PATH: '/home/ezra/CHIRPS_truth/'                                    # CHIRPS, pre-regridded
        FORECAST_PATH: '/home/ezra/IFS_training/'                                 # same IFS as RFE2 — do not re-fetch
        CONSTANTS_PATH: '/home/ezra/SEWAA-forecasts-RFE2/SEWAA-forecasts/cGAN_data/'
        NORMALISATION_PATH: '/home/ezra/CHIRPS_cGAN_run/'                         # CHIRPS-specific FCSTNorm2018.pkl
        LEAD_IDX: 1
    TFRecords:
        tfrecords_path: '/home/ezra/chirps_tfrecords/run02/'                      # separate from RFE2's — not interchangeable
```

> ⚠️ **`tfrecords_path` is global, not per-config.** It lives in `data_paths.yaml`, so it does
> *not* switch when you pass `--config config_run02.yaml`. Point it at the run you are about to
> train, or you will train run02's field set on run01's tfrecords — silently, for 24 h.
> `launch_train.sh` now echoes the resolved path before launching; read that line.

`config.yaml` (run01) / `config_run02.yaml` (run02) key settings — both start from the RFE2
champion run06, 14 fields, 64/256:

```yaml
log_folder: "../logs_CHIRPS_run01"     # or ../logs_CHIRPS_run02
trackio_project: "sewaa-cgan-chirps"   # separate project — CRPS scales are NOT comparable to RFE2
train_years: [2018, 2019, 2020]
val_years: [2021]                      # held out — never trained on, so this CRPS is a real skill number
```

---

## 2. Decisions already made — don't relitigate

- **14 fields, 64/256 architecture, from the start.** RFE2's sweep already
  answered field-set vs. architecture vs. batch-size (fields ~14% CRPS, arch ~3%, batch ~0%) —
  that ranking is a property of the model/IFS input, not the truth, so CHIRPS starts directly
  from run06's config instead of re-climbing the ladder.
- **run02 swaps `cape` for CHIRPS climatology, one-for-one.** `climatology.nc` supplies two
  variables (`climatology_mean`, `climatology_sd`), which occupy exactly the 2-channel slot
  `cape` vacated — so the channel count (28) and the architecture are byte-identical to run01.
  run02 vs run01 is therefore a clean controlled experiment on the field set alone. Since
  the sweep says field set dominates, this is the highest-value knob left.
- **The climatology field cannot leak the held-out year.** All four years' `climatology.nc`
  are built from `train_years=[2018,2019,2020]` only, so 2021's climatology never sees 2021.
  Within the training years it is leave-one-out: the target date's own truth is dropped by
  date equality before the mean/sd, so a training sample can't read its own answer. Verified
  from the files' `description` attribute — re-check it if the field is ever regenerated.
- **2021 is held out.** Train on 2018–2020, validate on 2021. A 2022+ held-out year is
  impossible — the IFS training archive stops at 2021. This costs a quarter of the training
  data vs. RFE2's champion (all 4 years), so run01's CRPS is not apples-to-apples with run06's.
- **Never compare CRPS to RFE2.** Different truth product, different scale/calibration.
  The only defensible baseline is a CHIRPS climatology on the same held-out year (§6).
- **Only the rain bins get recalibrated** (`tfrecords_generator.py:185`); everything else in
  the tfrecords/training code is untouched.
- **Pre-existing `/home/ezra/CHIRPS/` (Earth Engine) data is unusable** — 58.9% domain
  coverage, 89 truncated files, wrong years. Don't regrid it; it was re-downloaded from
  scratch instead (§3).

---

## 3. Pipeline already executed (for reference / rebuild)

```bash
cd /home/ezra/CHIRPS/cGAN_CHIRPS
PY=/home/ezra/cgan_env/bin/python

# 1. Truth — 1461 days, 2018-01-01..2021-12-31. Resumable; rerun freely.
$PY scripts/download_chirps_truth.py --start 2018-01-01 --end 2021-12-31 \
      --out /home/ezra/CHIRPS_truth
$PY scripts/audit_chirps.py --dir /home/ezra/CHIRPS_truth --pattern "*/*.nc"

# 2. IFS input — already there from the RFE2 project (/home/ezra/IFS_training/{2018..2021}).
#    Same ~400 GB, do not re-download.

# 3. Clone dsrnngan and drop in the CHIRPS patch.
cp -r /home/ezra/SEWAA-forecasts-RFE2/SEWAA-forecasts/24h_accumulations/cGAN \
      /home/ezra/CHIRPS_cGAN_run
cp dsrnngan_patch/data.py         /home/ezra/CHIRPS_cGAN_run/dsrnngan/data.py
cp dsrnngan_patch/data_paths.yaml /home/ezra/CHIRPS_cGAN_run/dsrnngan/data_paths.yaml
cp dsrnngan_patch/config.yaml     /home/ezra/CHIRPS_cGAN_run/dsrnngan/config.yaml

# 4. Rain bins — quartiles of raw mm/hr over random 128x128 crops of 2018 truth.
$PY scripts/calibrate_bins.py --truth /home/ezra/CHIRPS_truth --year 2018
#    -> paste result into dsrnngan/tfrecords_generator.py ~line 185
#       (already applied: bins = [0.0082, 0.0439, 0.1234])

# 5. FCSTNorm (14 fields, from 2018) — writes FCSTNorm2018.pkl
cd /home/ezra/CHIRPS_cGAN_run/dsrnngan
$PY run_gen_fcst_norm.py
#    must print "Loading forecast normalisations" on later imports, not "*** NOT LOADED ***"

# 6. TFRecords, per year 2018-2021
$PY -c "from tfrecords_generator import write_data; write_data(2018)"
$PY -c "from tfrecords_generator import write_data; write_data(2019)"
$PY -c "from tfrecords_generator import write_data; write_data(2020)"
$PY -c "from tfrecords_generator import write_data; write_data(2021)"
#    -> chirps_tfrecords/run01/{year}_1.{0,1,2,3}.tfrecords (19 GB total)
#    NOTE: these were DELETED on 2026-07-27 to make room for run02. run01's training and
#    eval are complete and do not need them (eval reads full-size images from raw netCDF,
#    not tfrecords). See §8 for how to rebuild if ever required.
```

---

## 4. Train — MUST detach from the JupyterHub cgroup

Launching from a JupyterHub browser terminal already killed this run twice (once via a
`jupyterhub.service` restart taking down its whole cgroup, `KillMode=control-group`). Use
`launch_train.sh` — it refuses to run from that cgroup and refuses if the GPU is already
occupied (a second `main.py` OOMs against the first without harming its checkpoint).

```bash
cd /home/ezra/CHIRPS_cGAN_run
./launch_train.sh                                       # fresh run01 (defaults to config.yaml)
./launch_train.sh --restart                             # resume run01 from the latest checkpoint
./launch_train.sh --config config_run02.yaml            # fresh run02
./launch_train.sh --config config_run02.yaml --restart  # resume run02
```

What it does that a bare `python main.py` would not:
- Refuses if the shell is inside `jupyterhub.service`'s cgroup (use a VS Code integrated
  terminal or `ssh ezra@e4drr-crafd-icpac` instead).
- Refuses if `main.py --config` is already running, or the GPU already holds >1 GB.
- Refuses if the named config doesn't exist or has no `log_folder`.
- Sets `LD_LIBRARY_PATH` to the venv's bundled NVIDIA libs — **without this, TF 2.21 silently
  falls back to CPU** and you lose the run to a no-op.
- Runs via `setsid nohup ... & disown`.
- **Names the log after the config's `log_folder`**, so two runs can never share a log file:
  `../logs_CHIRPS_run01` → `train_CHIRPS_run01.log`, `../logs_CHIRPS_run02` →
  `train_CHIRPS_run02.log`. Keyed on `log_folder` rather than the config filename because
  that's where `main.py` actually writes checkpoints — it's the authoritative run identity.
- **Echoes config / log_folder / tfrecords before launching.** Read the tfrecords line: it
  comes from the *global* `data_paths.yaml` and is the one setting `--config` does not switch.

Confirm it's actually on the GPU within a minute or two (substitute the run's own log):
```bash
grep -m1 'Created device' /home/ezra/CHIRPS_cGAN_run/dsrnngan/train_CHIRPS_run02.log
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

**Two log files exist — check which one the live PID actually holds before diagnosing
anything:**
```bash
ls -l /proc/$(pgrep -f '[m]ain.py --config')/fd/1
```
An earlier duplicate-process incident left `train_CHIRPS_run01.log` ending in a stale
`ResourceExhaustedError` from a dead process; some launches wrote to
`/home/ezra/chirps_run01_train.log` instead. Confirm liveness from GPU utilization and
`/proc/PID/stat` CPU jiffies (fields 14+15), never from a log file's mtime alone.

Perf: ~0.42 s/step, ~7.3 min/checkpoint → ~24 h for all 200 checkpoints.

---

## 5. Monitor — Trackio

```bash
source /home/ezra/cgan_env/bin/activate
trackio show --project "sewaa-cgan-chirps"
```
`config.yaml`'s `trackio_project: "sewaa-cgan-chirps"` keeps this run's metrics in a separate
project from RFE2's `sewaa-cgan-rfe2` / `sewaa-cgan-eval`, so the two CRPS scales can't be
misread against each other. `--restart` resumes the same Trackio run (`resume="allow"`)
instead of forking a new one.

Raw GPU check, no install needed:
```bash
watch -n 2 nvidia-smi
```

---

## 6. Evaluate

```bash
cd /home/ezra/CHIRPS_cGAN_run/dsrnngan
python main.py --config config_run02.yaml --no_train --evaluate --eval_full
```
Compare against a **CHIRPS climatology baseline**, never RFE2's numbers — and use the one
matching this eval's dates and ensemble size:

```
✅ USE THIS for a 25-date eval:
CHIRPS/cGAN_CHIRPS/climatology_baseline_2021_evaldates_10member.txt
  25 dates (the eval's own, seed 9999), 10-member ensemble
  CRPS 0.050696  (max_4 0.081905, avg_4 0.048478, EMRMSE 0.186088, MAE 0.079038)

⚠️ NOT this one, unless the model was also scored on a full year at 8 members:
CHIRPS/cGAN_CHIRPS/climatology_baseline_2021.txt
  364 dates (2021-01-02..12-31), 8-member ensemble
  CRPS 0.046074  (max_4 0.0716, avg_4 0.0440, EMRMSE 0.1709, MAE 0.0715)
```
Using the wrong one flips verdicts — see the warning immediately below.
Regenerate either: `CHIRPS/cGAN_CHIRPS/scripts/climatology_baseline.py` (CPU, ~35 s).

### ⚠️ NEVER compare a model CRPS to a baseline scored on different dates

This produced a **wrong published verdict once already** (run01 recorded as losing by 3.0%
when it actually wins by 6.4%). Two mismatches, in opposite directions:

| | Model eval | Full-year baseline |
|---|---|---|
| Dates scored | **25** (`EVAL.num_batches: 25`; `evaluation.py:103` fixes `batch_size = 1`, one full image per batch) | **364** |
| Ensemble members | **10** (`main.py:281` hardcodes `ensemble_size=10`, overriding `TRAIN.ensemble_size: 8`) | **8** |

The 25 dates are *harder than average*: climatology scores 0.046074 over the year but 0.050696
on those 25. April supplies 5 of the 25, February and November none. Comparing a model measured
on hard dates against a baseline averaged over easy ones manufactured a deficit that never
existed. This is a **bias, not noise** — more data on one side does not fix it.

**Use `climatology_baseline_2021_evaldates_10member.txt` (0.050696) for any 25-date model eval.**
The full-year 0.046074 figure is only comparable to a full-year model eval, and only after
re-running it at `--ensemble-size 10`.

Rebuild the matched baseline (CPU, seconds) with:
```bash
python /home/ezra/CHIRPS_cGAN_run/scripts/extract_eval_dates.py --config dsrnngan/config_run02.yaml
python scripts/climatology_baseline.py --ensemble-size 10 \
    --dates-file /home/ezra/CHIRPS_cGAN_run/scripts/eval_dates_2021.txt
```
`extract_eval_dates.py` reads the date list off the real DataGenerator rather than
re-implementing the seed-9999 shuffle, so it cannot drift. Cross-check: its first four dates
must match `eval_validation.txt`'s header line.

**Do NOT raise `num_batches` for the 200-checkpoint sweep.** 25 dates is fine for *ranking*
checkpoints — the seed is fixed (9999) and `evaluation.py:101` re-creates the iterator per
checkpoint, so every checkpoint and both runs see the same 25 dates. That makes run01-vs-run02
a properly **paired** comparison, where common date difficulty cancels. Only the headline-vs-
climatology number needs a full year (§9). Raising it would make the sweep ~14× slower for nothing.

### Both runs converge early — do not train longer than 200 checkpoints

run01's top 10 checkpoints span 116,736–202,752 samples with a 0.9% spread and no trend;
run02's best is at 181,248 with its whole back half at or below run01's peak. Both plateaued
around the midpoint, and run02 reached run01's *final* score by checkpoint ~67. Ordering within
the plateau is noise. Extra checkpoints buy nothing; **change the field set instead.**

`predict.py` works the same as RFE2:
```bash
python predict.py --log_folder ../logs_CHIRPS_run02 --model_number <ckpt> --num_samples 5
```

---

## 7. Gotchas specific to this run (don't re-debug these)

- **netCDF4/HDF5 is not thread-safe.** The truth downloader learned this the hard way: threaded
  netCDF reads segfaulted, and the resume logic then misread the crash as "corrupt file" and
  deleted good output. Fixed by routing all netCDF I/O through a single lock; keep that pattern
  if you touch the downloader.
- **CHC `africa_daily` is missing all of December 2021.** Not a bug — a real gap in that
  mirror. The downloader falls back to `global_daily` for those 31 days (verified identical
  land mask, correlation 1.000 on an overlapping day).
- **Units: CHIRPS is mm/day, IFS `tp` is mm/hr.** `load_truth_and_mask()` divides by 24;
  `calibrate_bins.py` mirrors that so the bins are computed in the same units training sees.
- **A second `main.py --restart` while the first already holds the GPU OOMs and exits in ~24 s**
  — too fast to reach a checkpoint save, so no weights are harmed, but it does leave a stale log
  tail that looks like the live run crashed (see §4's two-log-file note).
- **`FCSTNorm2018.pkl` is a single global file — changing the field set overwrites it.** run02's
  rebuild clobbered run01's. run01's copy is preserved at
  `FCSTNorm2018_run01_14field_cape.bak`. To go back to a run01 field set you must restore that
  file *and* revert `all_fcst_fields` in `data.py` (`climatology` → `cape`, in all three lists).
  Doing one without the other gives a silent normalisation mismatch, not an error.
- **`data_paths.yaml`'s `tfrecords_path` is global too** — see the warning in §1.
- **ZFS frees space asynchronously.** After deleting the 19 GB of run01 tfrecords, `df` still
  showed the old figure; it corrected after a `sync` and a few seconds. Don't panic-delete more.
  Check `zfs list -o usedbysnapshots tank/home/ezra` before assuming snapshots are holding data
  (they were not — `0B`).

---

## 8. run02 — the climatology-for-cape swap (COMPLETE, kept as the recipe)

✅ Finished 2026-07-29 and is now the champion (§0). Kept here as the worked recipe for the
next field-set experiment: a run03 varying the field set should follow exactly these steps,
changing only `all_fcst_fields` in `data.py`, the config's `log_folder`, and
`data_paths.yaml`'s `tfrecords_path`. **Both FCSTNorm and the tfrecords must be rebuilt
whenever the field set changes** — see §7.

The climatology field itself was generated by:

```bash
cd /home/ezra/CHIRPS_cGAN_run
/home/ezra/cgan_env/bin/python scripts/make_climatology_field.py
#   -> /home/ezra/IFS_training/{2018..2021}/climatology.nc  (~2 GB/year)
#   loaded through the ordinary load_fcst() path, so it needs no special-casing in data.py
#   beyond membership in all_fcst_fields / nonnegative_fields
```

Remaining steps, in order. Steps 1–2 are CPU only — the GPU must not be touched until step 3.

```bash
PY=/home/ezra/cgan_env/bin/python
cd /home/ezra/CHIRPS_cGAN_run/dsrnngan

# 1. Confirm FCSTNorm matches the run02 field set (should list 'climatology', NOT 'cape')
$PY -c "import pickle; print(sorted(pickle.load(open('../FCSTNorm2018.pkl','rb')).keys()))"

# 2. TFRecords for run02 — ~19 GB, one command per year
$PY -c "from tfrecords_generator import write_data; write_data(2018)"
$PY -c "from tfrecords_generator import write_data; write_data(2019)"
$PY -c "from tfrecords_generator import write_data; write_data(2020)"
$PY -c "from tfrecords_generator import write_data; write_data(2021)"
#   -> chirps_tfrecords/run02/{year}_1.{0,1,2,3}.tfrecords

# 3. Train (~24 h). NOT from a JupyterHub terminal — see §4.
cd /home/ezra/CHIRPS_cGAN_run
./launch_train.sh --config config_run02.yaml

# 4. Confirm it reached the GPU, within a minute or two
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# 5. Evaluate, then read §6's comparability warning BEFORE quoting a CRPS number
cd dsrnngan && $PY main.py --config config_run02.yaml --no_train --evaluate --eval_full
```

Disk: 60 GB free as of 2026-07-27 (after reclaiming run01's 19 GB), so step 2 leaves ~41 GB.

**Rebuilding run01's tfrecords**, if the deleted set is ever needed again: restore
`FCSTNorm2018_run01_14field_cape.bak` over `FCSTNorm2018.pkl`, revert `all_fcst_fields` in
`data.py` to the `cape` set, point `tfrecords_path` back at `run01/`, and re-run the
`write_data` loop. ~30 min CPU. The rain bins do **not** change — they are truth-derived.

---

## 9. Full-year confirmation — ✅ DONE 2026-07-29

**Result: run02 CRPS 0.041912 vs climatology 0.044954 over 364 dates at 10 members — a 6.8%
win, 7/7 metrics.** Full table in §0. This is the number to report.

The 25-date eval is enough to *rank* checkpoints and to settle run01-vs-run02 (paired, same
dates). It was a thin basis for a **reported** number: 25 of 365 dates, seasonally skewed.
A colleague's request — "a full year of forecasts from the trained model, compared to a full
year of climatological forecasts, ideally on a test year not in the training data" — was exactly
this gap. The two commands, ~24 min, rerunnable if a new champion appears:

```bash
# 1. Model, full year, best checkpoint only (~24 min GPU, 4 s/image)
cd /home/ezra/CHIRPS_cGAN_run && \
export LD_LIBRARY_PATH=$(ls -d /home/ezra/cgan_env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':') && \
setsid nohup /home/ezra/cgan_env/bin/python scripts/eval_fullyear.py \
  --config dsrnngan/config_run02.yaml --model-number 181248 \
  > logs_CHIRPS_run02/eval_fullyear.log 2>&1 < /dev/null & disown

# 2. Climatology, full year, SAME 10 members (CPU, ~35 s; safe to run concurrently)
cd /home/ezra/CHIRPS/cGAN_CHIRPS && /home/ezra/cgan_env/bin/python scripts/climatology_baseline.py \
  --ensemble-size 10 --out /home/ezra/CHIRPS/cGAN_CHIRPS/climatology_baseline_2021_10member.txt
```

Step 2 needs no `--out`: since 2026-07-29 the default filename encodes the eval year, the date
restriction and the ensemble size — `climatology_baseline_2021_10member.txt` here — and the
script **refuses to overwrite an existing file** unless given `--force`. Two runs with different
settings can no longer land on the same file, and a baseline that comparisons cite can't be
replaced by accident. (Before that fix, `--ensemble-size 10` with no `--out` silently
overwrote the 8-member `climatology_baseline_2021.txt`.)

`scripts/eval_fullyear.py` exists because **`main.py` cannot evaluate an arbitrary single
checkpoint** — `--eval_blitz` takes the last 4, `--eval_short` the last half, `--eval_full` all
200, and run02's best (181248) is in none of them. It calls `evaluate_multiple_checkpoints`
directly with an explicit `model_numbers` list, deriving every model parameter from the same
config `main.py` reads. It writes to `eval_fullyear.txt`, **not** `eval_validation.txt` — that
file records no per-row date count, so a 364-date row sitting among 25-date rows would be
silently misleading.

Gotcha already paid for: any standalone script that builds the model **must set
`TF_USE_LEGACY_KERAS=1` before importing TF** (as `main.py:5` does) and call
`read_config.set_gpu_mode()`. Without the first, model construction dies with
"A KerasTensor cannot be used as input to a TensorFlow function".

### On "a test year not in the training data"

2021 **is** excluded from training (`train_years: [2018,2019,2020]`), and the climatology field
is built from training years only, so it smuggles nothing in. But 2021 is also the year the
best checkpoint is *selected* on — so reporting 2021 CRPS for the argmax of 200 checkpoints is
mildly optimistic. 2021 is a **validation** year, not a clean test year.

A truly clean test year is **not currently possible**: IFS inputs and CHIRPS truth both stop at
2021 (verified 2026-07-29). `data.get_dates` asserts `year in (2018..2022)` with the comment
"2022 kept for held-out testing" — the code anticipates it, the data was never fetched.
Two ways to close it:

- **Split 2021** — select the checkpoint on one subset of dates, report on the disjoint rest.
  Free, no new data, gives an unbiased estimate at the cost of halving each side.
- **Acquire 2022** — the real fix. ~100 GB of IFS input against ~41 GB free, plus download time
  at the archive's 4–7 MB/s ceiling.

State this limitation whenever quoting the number, rather than letting "2021 was held out"
imply more than it does.

---

## Quick reference — command order

```text
# one-time / already done (truth side, shared by every run):
python scripts/download_chirps_truth.py --start 2018-01-01 --end 2021-12-31 --out /home/ezra/CHIRPS_truth
cp -r .../24h_accumulations/cGAN /home/ezra/CHIRPS_cGAN_run
cp dsrnngan_patch/{data.py,data_paths.yaml,config.yaml} /home/ezra/CHIRPS_cGAN_run/dsrnngan/
python scripts/calibrate_bins.py --truth /home/ezra/CHIRPS_truth --year 2018   # -> tfrecords_generator.py:185

# per run (field set changes => BOTH of these must be redone):
python run_gen_fcst_norm.py                                                    # -> FCSTNorm2018.pkl
python -c "from tfrecords_generator import write_data; write_data(Y)"          # Y in 2018..2021
#   ...and point data_paths.yaml's tfrecords_path at that run's folder first

# run02 only, already done:
python scripts/make_climatology_field.py                # -> IFS_training/{year}/climatology.nc

# GPU node, repeat as needed:
./launch_train.sh --config config_run02.yaml [--restart]
trackio show --project "sewaa-cgan-chirps"
python main.py --config config_run02.yaml --no_train --evaluate --eval_full   # after training completes
#   -> compare against climatology_baseline_2021_evaldates_10member.txt (0.050696), NOT the
#      full-year 0.046074. Wrong baseline flips the verdict — see §6.

# headline number for a writeup (§9):
python scripts/eval_fullyear.py --config dsrnngan/config_run02.yaml --model-number 181248
python scripts/climatology_baseline.py --ensemble-size 10    # -> ..._2021_10member.txt
```

---

## 10. El Niño hindcast — OND 2023 (built 2026-09-14, NOT yet run)

§9 closed the "full year, held-out" gap with 2021 — a *validation* year, not a clean test
year, since it is also the year every checkpoint was selected on. **OND 2023 closes the rest
of it**, and asks a different question.

**Why OND 2023.** Strong El Niño, the short-rains season ICPAC actually issues impact
advisories for, and genuinely out-of-sample twice over: outside `train_years
[2018,2019,2020]` *and* outside 2021. Its IFS input has been on disk since the RFE2 held-out
work (`IFS_training/2023`, 2023-06-01..12-31) — no download. §9's "a clean test year is not
currently possible" is **obsolete**: it was written before 2023/2024 IFS arrived.

**The question is not CRPS.** A model can beat climatology on CRPS while forecasting a
*normal* season during an El Niño — CRPS rewards hedging, exactly as it hid the heavy-rain
area deficit in [[extremes]]. So the headline here is **percent of normal, observed vs
hindcast**, read alongside the mean bias and two bias-invariant correlations (anomaly
pattern *r*, daily series *r*). Percent-of-normal alone moves one-for-one with the model's
own wet/dry bias and cannot separate "saw the El Niño" from "is wet".

**⚠️ The bug this depends on.** `data.load_fcst` indexed field files by *day of year*. The
2018-2021 files start on 1 Jan so that was right for them; **2023 starts on 1 Jun**, so
day-of-year read 2023-09-30 as index 272 instead of 121 — the wrong day, silently, no error.
Fixed 2026-09-14 by `data.fcst_time_index`, which resolves the date on the file's own time
axis and falls back to day-of-year for `climatology.nc` (whose bare 0..n-1 axis genuinely is
day-of-year). Full-year indices are unchanged — asserted by
`scripts/check_fcst_time_index.py`, which must stay passing. The RFE2 project hit and fixed
the same bug in Aug 2026; the CHIRPS copy of `data.py` had not been patched.

```bash
# 0. Confirm nothing already computed moved, and see the 2023 correction
python scripts/check_fcst_time_index.py

# 1. Preflight — prints the exact command for anything missing. Run this FIRST.
python scripts/hindcast_elnino.py --stage check

# 2. The two gaps it finds (both cheap):
CLIM_YEARS=2023 python scripts/make_climatology_field.py        # ~15 min CPU, ~2 GB
python /home/ezra/CHIRPS/cGAN_CHIRPS/scripts/download_chirps_truth.py \
    --start 2023-10-01 --end 2023-12-31                         # ~1 min, no login

# 3. Generate — GPU, ~30 min per run, ~0.9 GB each. MUST detach (§4).
cd dsrnngan && export LD_LIBRARY_PATH=$(ls -d /home/ezra/cgan_env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
setsid nohup python forecast_batch_crps.py forecast_run07_elnino_OND2023.yaml > ~/elnino_run07_OND2023.log 2>&1 < /dev/null & disown
setsid nohup python forecast_batch_crps.py forecast_run02_elnino_OND2023.yaml > ~/elnino_run02_OND2023.log 2>&1 < /dev/null & disown

# 4. Score — CPU, minutes
python scripts/hindcast_elnino.py --stage score     # -> ~/elnino_hindcast_OND2023/
```

**Both champions, deliberately.** run07 @ 204800 (extremes champion, 5-bin reweight) and
run02 @ 181248 (mean-CRPS champion). El Niño is an extremes question, but §6's rule stands:
never let one metric pick the winner.

**Two climatology baselines, say which you quote.** `clim3` = 2018-2020 ±7d, the project
convention and the only thing comparable with the CRPS numbers already in this runbook —
its pool is ~45 days, which **caps the scored ensemble at 45 members even though the model
writes 50**. `clim20` = 2002-2021 ±7d (~293 members), the harder, fairer baseline and the
anomaly reference; it excludes 2023 so it cannot see the event. Model and both baselines are
scored on the same days at the same *n* — see §6.

**Do not select anything on this season's tail metrics.** One season of rare events is a
winner's curse. Select on CRPS elsewhere, *report* the tail numbers here.
