# ifs-0p1-gik: ECMWF IFS ENS at 0.1° to cGAN inference inputs over East Africa

This folder runs the full path from the ECMWF open-data GRIB files to the NetCDF file the cGAN inference reads:

1. Build a **virtual** Icechunk store from the GRIB `.index` files.
2. Optionally publish that store to source.coop.
3. Start Frisky workers on the EWC VMs.
4. **Realize** the 14 cGAN fields over the East Africa box into a real Icechunk store.
5. Check the realized store.
6. Optionally publish the realized store to source.coop.
7. Write `IFS_YYYYMMDD_HHZ.nc` on the TF cGAN's own 384 × 352 frame.
8. Run the cGAN inference on it (`step7_infer.py`) and optionally upload the output; the IFS NetCDF is then deleted.

The workflow runs in two modes:

| Mode | Where it runs | What it uses |
|---|---|---|
| **local** (`run_local.sh`) | One machine | Local directories only. **No source.coop and no Ceph.** |
| **ewc** (`run_ewc.sh`) | EWC Frisky workers | Writes to Ceph, publishes to source.coop, builds the NetCDF from the public copy, runs the inference. This is the daily driver. |

The code was moved here from `grib-index-kerchunk` (branch `ecmwf-0p1-survey`, `1d12c38`). It comes from:
- `ecmwf/gcs-source`
- `ecmwf/icechunk-par`
- `ecmwf/0p1/realize-0p1`
- `ecmwf/0p1/wide-box/realize`

It has been cut down to this one box and these 14 fields; §12 lists what came from where.

---

## 0. Measured

### 2026-10-27, box 19–55°E, seven cycles 20261020T18z … 20261026T00z (ewc mode)

| Cycle | Realize (5 VMs × 6 tasks) | Saved by prefix read | Store | Publish | TF-frame NetCDF | step 7 dry run |
|---|---|---|---|---|---|---|
| 20261020T18z (25 lead times) | 266 s, 67.0 msg/s | 32.6% | 2.17 GB | 23 s | 2.31 GB, 218 s | OK, 24 windows |
| 20261021T00z | 330 s, 62.8 msg/s | 32.4% | 2.56 GB | 33 s | 2.72 GB, 310 s | OK, 28 windows |
| 20261022T00z | 302 s, 68.5 msg/s | 32.3% | 2.58 GB | 58 s | 2.75 GB, 287 s | OK |
| 20261023T00z | 328 s, 63.1 msg/s | 32.3% | 2.58 GB | 64 s | 2.75 GB, 263 s | OK |
| 20261024T00z | 425 s, 48.8 msg/s | 32.2% | 2.58 GB | 67 s | 2.76 GB, 250 s | OK |
| 20261025T00z | **2363 s, 8.8 msg/s** | 32.0% | 2.59 GB | 36 s | 2.77 GB, 480 s | OK |
| 20261026T00z | **1553 s, 13.3 msg/s** | 31.8% | 2.59 GB | 56 s | 2.76 GB, 395 s | OK |

- Every cycle passed step 5 on Ceph (406/406 chunks, 350/350 for 18z, 12/12 bit-exact) and again on source.coop, read anonymously.
- **The two slow cycles were the GRIB reads, not this setup.** Worker memory stayed at 1.5–1.9 GB per VM and CRMA was not running, but the mean GET time doubled (3.4 s against 1.6 s), and the last 86 of 406 blocks took 22 min. That looks like AWS throttling or slow servers on ECMWF's side.
- **NetCDF size.** On the TF frame the file is ~2.75 GB, more than on the store grid (2.36 GB): the four-point mean makes the values compress worse. The HDF5 shuffle filter did not help.
- **Inference.** Tested with an untrained generator (`--untrained`, the GEFS v3 `setup_params.yaml` as the stand-in architecture): 2 windows × 4 samples in 16 s on 8 CPU cores, output 384 × 352 on the frame. A full cycle (28 windows × 50 samples) extrapolates to ~30 min on CPU. The trained IFS weights are not on this machine (§8).

### 2026-10-27, first trial, box 20–53°E, cycle 20261021 00z

| | local mode (small trial) | ewc mode (full cycle) |
|---|---|---|
| Virtual store, one cycle | 742,478 refs from 340 `.index` files, 44 s, 17 MB; 8/8 read-back equal | (public store, snapshot `6AK8YKP0TDZJ085CV960`) |
| Realize | 14 fields × 4 steps × 6 members = 336 msgs, 12 s, 28 msg/s, gateway 2 × 4 | **14 × 29 × 51 = 20,706 msgs in 325 s, 63.8 msg/s**, 5 VMs × 6 tasks, 0 failed blocks |
| Bytes fetched | — | **28.4 of 42.1 GB (32.5% saved)** by the prefix read; 5 top-up GETs |
| Worker memory | — | VM used ≤ 1.8 GB of 16 GB |
| Realized store | — | **2.35 GB per cycle**, 436 objects |
| Check (step 5) | PASS: 56/56 chunks, 8/8 bit-exact | PASS on Ceph: 406/406 chunks, 12/12 bit-exact. PASS on source.coop (anonymous): 4/4 |
| Publish to source.coop | not used | 2.35 GB in 25 s (95 MB/s), 0 errors |
| NetCDF (step 6) | 5 members, 4 lead times, 0.03 GB, 2 s | from source.coop, anonymous: 14 fields × 29 lead times × **10 members, 0.43 GB in 404 s** (first field 259 s, cold; then ~10 s per field; a repeat read of one field takes 9.5 s). The full 50-member file extrapolates to **~2.2 GB** |

---

## 1. What is built

**Box.** lat 25°N → 15°S and lon 19°E → 55°E at 0.1°, which is **401 × 361** points. Every edge lies on the 0.1° grid (origin −180°, 90°N), so the cut is exact. It covers the TF cGAN training frame (384 × 352 at −13.65 … 24.65°N, 19.15 … 54.25°E) with a margin on every side for interpolating onto it. The first trial used 20 … 53°E (`20e53e-25n15s`), which did not cover that frame; that store is superseded.

**Step-6 output frame.** The TF cGAN does not read latitude/longitude from its input: it assumes the 384 × 352 frame at −13.65 … 24.65°N, 19.15 … 54.25°E. `step6_make_cgan_netcdf.py --frame tf` (the default) writes that frame. Each frame point lies exactly halfway between four 0.1° points (x.x5), so it is the mean of those four, which is bilinear interpolation at a cell centre. `elev.nc`, `lsm.nc` and `FCSTNorm2018.pkl` are on the same frame, ascending latitude.

**Fields.** These are the 14 cGAN forecast fields (`data.all_fcst_fields`, see `RFE2_cGAN_RUNBOOK.md`):

| cGAN name | `.index` param | level | notes |
|---|---|---|---|
| `cape` | `mucape` | sfc | ECMWF publishes most-unstable CAPE under `mucape` since 2024-11-13. There is no `cape` in the 0.1° files. |
| `cp`, `tp` | `cp`, `tp` | sfc | m, **accumulated from the init** |
| `ssr` | `ssr` | sfc | J m⁻², **accumulated from the init** |
| `mcc` `sp` `tciw` `tclw` `tcrw` `tcw` `tcwv` | same | sfc | |
| `t2m` | `2t` | sfc | |
| `u700`, `v700` | `u`, `v` | 700 hPa | |

**Lead times.** 0 … 168 h every 6 h, which is 29 steps. These are the `valid_time`s that `ensemble-forecast.py` steps through with `HOURS = 6`. All of them exist in both the 00z and 12z runs. The 06z and 18z runs stop at 144 h, so their stores and NetCDFs get the **25** lead times 0 … 144 h; `step4_realize.py` trims them and `step6` warns.

**Members.** All 51: the control (`number = 0`, from `oper/fc`) and the 50 perturbed members (from `enfo`). Step 6 chooses which go into the NetCDF.

Everything above is set in **`config.py`**, which is the only place to change it.

---

## 2. Data flow

```
 ECMWF open data (same bytes in both)
   s3://ecmwf-forecasts/<date>/<HH>z/ifs/0p1/{enfo,oper}/...-{pl,sfc}.grib2  (+ .index)
   gs://ecmwf-open-data/   (same paths)
        |
        |  step1_build_virtual.py: reads only the .index files (340 per cycle)
        v
 VIRTUAL store  (Icechunk, references only, ~13-17 MB per cycle)
   local:  ~/icechunk-local/ecmwf-ifs-0p1-icechunk-vd
   public: source.coop forecasts/ecmwf-ifs-0p1-icechunk-vd    <- step 2, optional
        |
        |  step4_realize.py: coordinates and provenance from the virtual store;
        |  the bytes come from one Range GET of each message's prefix,
        |  decoded with libaec (cgan_dag.read_prefix)
        v
 REALIZED store  (Icechunk, real float32 values, 2.35 GB per cycle, one store per cycle)
   local:  ~/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z
   EWC:    ceph  must-icechunk/cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z
   public: source.coop forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z
        |
        |  step5_check.py: complete, bit-exact, sane
        |  step6_make_cgan_netcdf.py --frame tf   (384 x 352, 4-point mean)
        v
 IFS_<YYYYMMDD>_<HH>Z.nc  (temporary, ~2.75 GB; deleted after step 7)
        |
        |  step7_infer.py   (TF 2.15, trained generator weights)
        v
 GAN_IFS_<YYYYMMDD>_<HH>Z.nc  precipitation mm/h, (time, member, valid_time, 384, 352)
        |
        |  UPLOAD_GAN=1 only (off for now): upload_files.py
        v
 source.coop forecasts/ecmwf-ifs-0p1-cgan-gan6h/<YYYYMMDD>T<HH>z/
```

What each step reads and writes, by mode:

| Step | local mode | ewc mode |
|---|---|---|
| 1 virtual store | `.index` from GCS → local dir | same, then optionally step 2 |
| 2 publish virtual | — | local dir → source.coop (`step1_append_when_published.sh` with `PUBLISH=1`) |
| 3 workers | — | `step3_deploy_workers.py` |
| 4 realize | local Frisky cluster, local virtual store → local dir | EWC Frisky workers, public virtual store → Ceph |
| 5 check | local | Ceph, then the source.coop copy, anonymously |
| publish realized | — | Ceph → source.coop (`publish_ceph_to_source_coop.py`) |
| 6 NetCDF | from the local dir | from source.coop, anonymously; or from Ceph or local |
| 7 inference | `step7_infer.py` on the NetCDF | same; output uploaded only with `UPLOAD_GAN=1` |

**source.coop is never required.** It is the public copy, for other people and other machines. Every script also takes `local:` and `ceph:` stores (`stores.py`). The only network dependency the local mode keeps is ECMWF's own buckets, which it reads anonymously.

---

## 3. Setup

```bash
cd cGAN_tutorial/ifs-0p1-gik
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

The pins match the worker venv that step 3 builds on each VM. `imagecodecs` supplies libaec for the prefix decode.

Step 7 needs TensorFlow 2.15 (the version the TF cGAN was trained with, `env/env-tf.yaml`), which needs Python ≤ 3.11, so it has its own venv (~1.3 GB):

```bash
uv venv .venv-infer --python 3.11
uv pip install --python .venv-infer/bin/python -r requirements-infer.txt
```

Credentials are needed only in EWC mode. They are read from files and never printed.

| For | What | Used by |
|---|---|---|
| Ceph | `KEY=VALUE` file with `AK=` and `SK=` (`CEPH_ENV`) | step 4 sink, step 5, publish |
| source.coop writes | `.env.sourcecoop` (`SC_ENV`) plus the `source-coop` CLI (`source-coop creds --format env`) | step 2, publish |

Reading from source.coop needs **no** credentials.

**The `AWS_*` trap.** icechunk caches its object-store configuration for the whole process. A Ceph `AWS_ENDPOINT_URL` left in the environment would redirect the anonymous reads of `s3://ecmwf-forecasts`. So:
- `cgan_dag.py` deletes every `AWS_*` variable on import, and every script imports it first;
- the workers are started without them;
- the publish scripts receive the source.coop STS credentials in a subshell only.

---

## 4. Step by step

### Step 1: virtual store (`step1_build_virtual.py`)

```bash
.venv/bin/python step1_build_virtual.py --date 20261021 --run 00 --resolution 0p1 \
    --store ~/icechunk-local/ecmwf-ifs-0p1-icechunk-vd --verify 12
# or wait for publication, append, and optionally publish:
bash step1_append_when_published.sh 2026102700                 # local only
PUBLISH=1 SC_ENV=... bash step1_append_when_published.sh 2026102700
```

- **Inputs.** It reads the 340 `.index` files of one cycle from GCS: 85 steps × {enfo, oper} × {pl, sfc}.
- **Commit.** It writes all 742,478 references (all 61 variables, not only the 14) as one commit to group `0p1/<HH>z`.
- **Verification.** It reads N random chunks back through the store and compares each with a direct gribberish decode.
- **No templates.** Nothing from the old pars/template pipeline is needed. `grids.py` and `build_ecmwf_icechunk.py` sit next to it. `grids.py` is **byte-identical** to `grib-index-kerchunk/ecmwf/grids.py`, and the builder refuses to run if two copies it can reach differ. That is the guard against the 180° longitude bug.
- **When the data appears.** 0.1° files appear about 2 h after the 0.25° files of the same run. `step1_append_when_published.sh` polls for the last step's four `.index` files every 5 min, for up to 10 h.

### Step 2: publish the virtual store (optional)

`publish_local_to_source_coop.py` copies the local store directory to `forecasts/ecmwf-ifs-0p1-icechunk-vd`:
- the copies are plain object copies;
- icechunk objects are immutable, so objects already there at the same size are skipped;
- the `repo` pointer goes up **last**, so a reader never sees a pointer to missing objects.

### Step 3: EWC workers (`step3_deploy_workers.py`), EWC mode only

```bash
D=/opt/mamba/envs/dask/bin/python          # has `distributed`; the Dask cluster is only a remote-exec channel
$D step3_deploy_workers.py --install       # venv in /tmp/frisky-cgan-ea on every VM (~40 s)
$D step3_deploy_workers.py --scheduler-on 192.168.1.164 --start --nthreads 6 --memory-limit 12GB
$D step3_deploy_workers.py --status
$D step3_deploy_workers.py --stop          # this workflow's workers and scheduler only
```

- **Isolated.** The workers run in their own directory (`/tmp/frisky-cgan-ea`) on their own ports: scheduler **8798**, dashboard 8793. The CRMA daily routine (`/tmp/frisky-ea`, :8796) is never touched. `--stop` matches only this directory's processes.
- **No credentials.** The workers are started with `AWS_*` scrubbed from *their* environment only.
- **One file shipped.** `cgan_dag.py` is the only file the workers need. It imports nothing from this folder; the box and field table arrive as task arguments. So changing `config.py` needs no redeploy.
- **Not alongside CRMA.** Don't run this while the CRMA daily run is active; both use every VM. `run_ewc.sh` waits for it.

### Step 4: realize (`step4_realize.py`)

```bash
# local
.venv/bin/python step4_realize.py --date 20261021 \
    --vd local:$HOME/icechunk-local/ecmwf-ifs-0p1-icechunk-vd \
    --sink local:$HOME/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261021T00z \
    --workers 2 --threads 4
# EWC
.venv/bin/python step4_realize.py --date 20261021 --sink ceph:cgan14-19e55e-25n15s/20261021T00z \
    --env $CEPH_ENV --scheduler 192.168.1.164:8798
```

What it does:

1. **Opens the virtual store.** It checks that the cycle and all 29 steps are there. It takes the box's latitude/longitude from the store and records the store's **snapshot ID** in the output, so you can always see which version of the store a realization came from.
2. **Lays down the schema.** One commit sets up the 14 arrays, each `float32 (time=1, step=29, number=51, latitude=401, longitude=361)` (step=25 for 06z/18z). A chunk is one (field, step) block, 29.5 MB raw. The **fill is NaN**, so a block that was never written can't pass as data.
3. **Submits one task per block** (406 tasks). Each `cgan_dag.read_write_block`:
   - reads its 51 messages with 4 threads into one preallocated array;
   - **writes the chunk from the same worker** into a shared icechunk *fork*;
   - returns the fork and exact byte counts.

   A sliding window (`--max-inflight 40`) bounds what is queued. Failed blocks are resubmitted in up to 4 rounds.
4. **Commits once.** It merges all forks into **one commit**, then adds `realize_summary` (bytes, rate) to the root attributes.

**The prefix read** (`cgan_dag.read_prefix`). This is a stand-in until gribberish can decode the first N rows of a message.
- **Why a prefix holds the box.** ECMWF packs 0.1° fields with CCSDS (libaec), rows north → south, and Section 7's first block starts at bit 0. So rows 90°N … 15°S are a **prefix** of each message.
- **The read.** One HTTP Range GET from the message start, decoded with libaec, gives the box exactly. gribberish 1.6.0 rejects a truncated message, so the GRIB2 header walk and the libaec call live in `cgan_dag.py` for now.
- **Byte offsets.** They come from the same GCS `.index` files step 1 used (icechunk 2.1 has no public call that returns a virtual reference).
- **AWS first, GCS fallback.** Data GETs go to AWS first. On a `503 Slow Down` they go to the byte-identical GCS copy.
- **Sizing the first GET.** Each worker learns how much of each field's message the box needs, from libaec's consumed-byte count. Its first GET for that field is the largest share seen so far plus 1.5%. A short guess costs a top-up GET, never a wrong value: 5 top-ups in 20,706 messages.
- **Saving.** 32.5% on this cycle. The rows kept are 58% of the message, but each field's compressed bytes are not spread evenly by row, and the first messages of each field on each worker start from a cautious guess.
- **Exactness.** Bit-exact, re-checked on every run by step 5 (and 736/736 against eccodes and gribberish in `grib-index-kerchunk/ecmwf/0p1/wide-box/README.md` §1).

`--full-read` reads whole messages through the virtual store and gribberish instead: slower, but with no libaec step.

**Why one task per block, not one per message.** With separate read tasks, every member's box lands on whichever worker read it, and the writer then pulls all of them across the LAN. On the wide box that limited the run to 2.6–5.7 msg/s instead of 16 msg/s (`grib-index-kerchunk/ecmwf/0p1/wide-box/README.md` §4).

### Step 5: check (`step5_check.py`)

It checks three things:

1. **complete:** every (field, step) chunk is in the manifest. A missing chunk reads as NaN, not as an error, so this has to be asked.
2. **exact:** `--samples` random (field, step, member) boxes are compared with a whole-message read through the virtual store. They must be **equal**, not close.
3. **sane:** all values are finite; members differ; `tp`, `cp` and `ssr` are 0 at step 0 and never decrease with lead time.

It exits non-zero on any failure, and `run_ewc.sh` publishes nothing unless it passes.

### Publish the realized store

`publish_ceph_to_source_coop.py` copies Ceph to source.coop, sending the `repo` pointer last. It uses multipart upload above 64 MiB because source.coop refuses single PUTs over ~100 MB; chunks here are smaller than that. For a local store, use `publish_local_to_source_coop.py --src <dir> --dest forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<cycle>`.

### Step 6: cGAN NetCDF (`step6_make_cgan_netcdf.py`)

```bash
.venv/bin/python step6_make_cgan_netcdf.py \
    --store sourcecoop:forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261021T00z --outdir nc
```

It writes `IFS_20261021_00Z.nc` in the layout **`scripts/ensemble-forecast.py`** reads, on the model's 384 × 352 frame (`--frame tf`, default; `--frame native` keeps the 401 × 361 store grid):

| variable | dims | content |
|---|---|---|
| `time` | (time=1) | init, hours since 1900-01-01 00:00:00.0 |
| `valid_time` | (valid_time=29; 25 for 06z/18z) | init + 0 … 168 h, hours since 1900-01-01 |
| `member`, `latitude` (ascending), `longitude` | | |
| `<field>_ensemble_mean` × 14 | (time, member, valid_time, latitude, longitude) | **every member**, float32, zlib 1 |

- **Variable name.** Despite the name, `_ensemble_mean` holds the members. That is the name the inference reads (`nc_in[f"{field}_ensemble_mean"][0, :, i:i+2]`), and the inference takes the mean and sd over members itself.
- **Accumulated fields.** `tp`, `cp` and `ssr` stay accumulated; the inference differences consecutive valid times.
- **Members.** `--members` defaults to `perturbed` (1 … 50, the 50 the script's comments describe). Use `all` to add the control.
- **Memory.** It reads one field at a time, as 29 chunks fetched `--threads` (6) at once, so memory peaks at ~0.8 GB.
- **Size.** 14 × 50 × 29 × 384 × 352 float32 is 11 GB raw, and **~2.75 GB** with zlib 1 (§0). `--members N` and `--fields` cut it down for tests.
- **No averaging needed beforehand.** The inference takes the ensemble mean and sd itself, and differences the accumulated fields.

### Step 7: cGAN inference (`step7_infer.py`)

```bash
.venv-infer/bin/python step7_infer.py --nc nc/IFS_20261021_00Z.nc \
    --model-folder /path/ICPAC-big-ensmeansd --checkpoint 316800 \
    --constants ../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS --outdir gan
.venv/bin/python step7_infer.py --nc nc/IFS_20261021_00Z.nc --constants ... --dry-run   # no TF
```

- **The same computation as `scripts/ensemble-forecast.py`.** Per 6-hour window and field it builds the 4 channels (mean/sd at start and end, or mean/sd of the accumulation), applies the same normalisation with `FCSTNorm2018.pkl`, adds `elev`/`lsm`, and runs the generator `--ensemble-members` times (50) with fresh noise. Output `GAN_IFS_<YYYYMMDD>_<HH>Z.nc`, `precipitation` in mm h⁻¹, `fcst_valid_time` = start of each window, as the original writes it.
- **Differences from the original script.** Paths are arguments, not `forecast.yaml`/`data_paths.yaml`. The input's frame is checked first. Members are predicted in batches (`--batch 10`). It loads only `model/models.py` and `model/noise.py` from `tensorflow-dev-test`, not `model/__init__.py`, which pulls in the training stack (cartopy, data paths); `keras.layers._Merge`, gone from Keras 2.15's top level, is aliased.
- **`--end-hour`** defaults to the file's last valid time (168 h; 144 h for 06z/18z).
- **`--dry-run`** builds and checks every network input (finite, 1 × 384 × 352 × 56) without TensorFlow or weights. `run_ewc.sh` falls back to it when `MODEL_FOLDER`/`CHECKPOINT` are unset.
- **`--untrained`** is for testing only: it builds the generator without loading weights.

### Upload the inference output (off for now)

`run_ewc.sh` with `UPLOAD_GAN=1` copies `GAN_IFS_*.nc` to source.coop `forecasts/ecmwf-ifs-0p1-cgan-gan6h/<YYYYMMDD>T<HH>z/` with `upload_files.py` (multipart, size-checked; `--to ceph:<prefix>` also works). Without it the output stays in `gan/`.

---

## 5. Running it

```bash
# everything on this machine, no source.coop
bash run_local.sh 20261021
STEPS="0 6 12" MEMBERS=5 NC_MEMBERS=4 bash run_local.sh 20261021      # small trial

# EWC + Ceph + source.coop: the daily run
APPEND_VD=1 CEPH_ENV=~/.../.env SC_ENV=~/.../.env.sourcecoop \
  MODEL_FOLDER=/path/ICPAC-big-ensmeansd CHECKPOINT=316800 TFPY=$PWD/.venv-infer/bin/python \
  bash run_ewc.sh 20261027
# several cycles in one go (workers started once)
CEPH_ENV=... SC_ENV=... bash run_ewc.sh 2026102018 2026102100 2026102200
# cycles already realized and published: NetCDF + inference only
SKIP_REALIZE=1 SC_ENV=... bash run_ewc.sh 20261021
```

`run_ewc.sh` env switches: `UPLOAD_GAN=1` uploads the inference output (off by default), `KEEP_NC=1` keeps the IFS NetCDF (deleted after step 7 by default), `CONSTANTS` (default `../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS`), `NC_MEMBERS`, `GAN_MEMBERS`.

The order inside `run_ewc.sh`: realize and check every cycle with the workers up, stop the workers, publish and check every cycle (~30–60 s each), then NetCDF + inference per cycle. Publishing first means one slow NetCDF does not hold back the public stores.

**Disk needed in local mode:**
- ~17 MB per cycle of virtual store;
- 2.35 GB per realized cycle;
- ~2.75 GB per full 50-member NetCDF on the TF frame, temporary;
- 1.3 GB for `.venv-infer`.

The gateway has only a few GB free, so `run_ewc.sh` keeps one NetCDF at a time and deletes it after step 7.

---

## 6. Where everything is stored

| What | local | EWC Ceph (`must-icechunk`) | source.coop (`e4drr-project`) |
|---|---|---|---|
| virtual store | `~/icechunk-local/ecmwf-ifs-0p1-icechunk-vd` | — | `forecasts/ecmwf-ifs-0p1-icechunk-vd` |
| cGAN inference output (`UPLOAD_GAN=1`) | `gan/GAN_IFS_<YYYYMMDD>_<HH>Z.nc` | — | `forecasts/ecmwf-ifs-0p1-cgan-gan6h/<YYYYMMDD>T<HH>z/` |
| realized, one store per cycle | `~/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z` | `cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z` | `forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z` |

Published realized cycles: `20261020T18z`, `20261021T00z` … `20261026T00z` under `forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/`. The first trial's 20–53°E store has been deleted from source.coop and Ceph. Opening one anonymously:

```python
import icechunk as ic, xarray as xr
st = ic.s3_storage(bucket="e4drr-project", endpoint_url="https://data.source.coop",
                   prefix="forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261021T00z",
                   region="us-east-1", anonymous=True, from_env=False, force_path_style=True)
ds = xr.open_zarr(ic.Repository.open(st).readonly_session("main").store, consolidated=False)
ds.tp.isel(step=4).mean("number")        # 24 h accumulated, m
```

---

## 7. Retention

**One store per cycle** is deliberate: deleting a cycle is a prefix delete. Deleting one date inside a multi-date icechunk store would need `expire_snapshots` and `garbage_collect`.

Once the NetCDF (or a downstream product) is made and checked, the realized cycle can go. It can be **recreated exactly**:
- the virtual store keeps the references;
- the GRIB files stay in ECMWF's buckets;
- the realized store's attributes record the virtual-store snapshot and the read method;
- recreating takes ~6 min of the EWC cluster per cycle.

Keep the virtual store; it is ~17 MB per cycle.

The IFS NetCDF is an intermediate: it is rebuilt from the realized store in ~5 min, so `run_ewc.sh` deletes it once the inference has run, and nothing keeps it on Ceph.

---

## 8. Caveats

- **The trained IFS generator weights are not on this machine.** The 14-field 6 h model (`ICPAC-big-ensmeansd`, `gen_weights-0316800.h5` and its `setup_params.yaml`) is on the SEWAA notebook server (`gefs-gik-data-prepration/cgan_ecmwf/ECMWF_cGAN_INFERENCE_WORKFLOW.md`). Only a GEFS model is in `tensorflow-dev-test/cgan_compact_20260202`. Until the IFS folder is copied here, step 7 runs as `--dry-run`.
- **`scripts/ensemble-forecast.py`** had two typos (`sys.path.insert("../data/")` without its index, `fron config import`), now fixed. It still imports `data_old`, which is not in this repo; `step7_infer.py` is the runnable equivalent.
- **Normalisation.** `FCSTNorm*.pkl` must come from the same field set and grid as the model.
- **`cape` is `mucape`.** It is the same channel as the model's `cape` from 2024-11-13; before that, ECMWF published `cape`.
- **06z/18z** runs stop at 144 h, so they can't provide the 150–168 h lead times: their NetCDF has 25 valid times, not 29. `step7_infer.py` stops at 144 h for them by itself; `ensemble-forecast.py` needs `end_hour: 144`.
- **The prefix read is a stand-in for a gribberish feature.** It assumes grid template 3.0, packing 5.42 (CCSDS), no bitmap and scan mode 0, and raises on anything else. Every 0.1° field checked so far meets those conditions.
- **The local Frisky cluster** must be reached with `cluster.get_client()`. `frisky.Client(cluster)` raises, and the worker processes then keep the interpreter alive.

---

## 9. Files

| File | Role |
|---|---|
| `config.py` | Box, 14 fields, lead times, store names: **the one place to edit** |
| `stores.py` | Store specs `local:` / `ceph:` / `sourcecoop:`, open helpers |
| `cgan_dag.py` | Worker module: prefix read, virtual read, `read_write_block` |
| `step1_build_virtual.py` | `.index` → virtual Icechunk store (with `build_ecmwf_icechunk.py`, `grids.py`) |
| `step1_append_when_published.sh` | Wait for a cycle, append, optionally publish (step 2) |
| `step3_deploy_workers.py` | EWC Frisky workers: install / start / status / stop |
| `step4_realize.py` | Realize one cycle (local cluster or EWC) |
| `step5_check.py` | Complete, bit-exact, sane |
| `step6_make_cgan_netcdf.py` | Realized store → `IFS_YYYYMMDD_HHZ.nc` (TF frame by default) |
| `step7_infer.py` | cGAN inference → `GAN_IFS_YYYYMMDD_HHZ.nc` (`requirements-infer.txt`, `.venv-infer`) |
| `publish_local_to_source_coop.py`, `publish_ceph_to_source_coop.py` | Copy a store up, `repo` last |
| `mem_monitor.py` | EWC: per-VM memory every N s |
| `run_local.sh`, `run_ewc.sh` | End-to-end drivers (`run_ewc.sh` takes several cycles) |
| `upload_files.py` | Copy finished files (the inference output) to source.coop or Ceph |
| `logs/` | Logs of the runs measured in §0 |

## 10. Provenance

All of the following come from `grib-index-kerchunk` at `1d12c38` (branch `ecmwf-0p1-survey`):

| Here | From |
|---|---|
| `step1_build_virtual.py` | `ecmwf/gcs-source/index_to_icechunk.py`, import path changed |
| `build_ecmwf_icechunk.py` | `ecmwf/icechunk-par/` |
| `grids.py` | `ecmwf/grids.py`, identical |
| `step1_append_when_published.sh` | `ecmwf/0p1/realize-0p1/append_when_published.sh` |
| `publish_*` | `ecmwf/0p1/realize-0p1/mirror_local_to_source_coop.py` and `wide-box/realize/mirror_ceph_to_source_coop.py` |
| `step3_deploy_workers.py` | `wide-box/realize/deploy_workers.py` |

`cgan_dag.py` and `step4`–`step7` are new; `step7_infer.py` follows `scripts/ensemble-forecast.py`. They are written for this box and field set, after:
- `wide-box/realize/frisky_daily_dag.py`;
- `wide-box/realize/build_corpus.py`;
- `wide-box/realize/sink_icechunk.py`.
