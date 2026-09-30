# RFE2 cGAN — CRPS Baselines & Dataset/Variables (runs 04 & 05)

Companion to **[RFE2_cGAN_RUNBOOK.md](RFE2_cGAN_RUNBOOK.md)**. Two purposes:

1. **Baseline-CRPS comparison** — put the run 04/05 CRPS numbers next to the
   baselines you must beat, and give the exact recipe to compute those baselines
   (there is no universal "ideal" CRPS — it only means something relative to a
   reference forecast).
2. **Dataset & variables: planned vs actual** — the runbook is written for the
   **14-field** plan, but runs 04 and 05 actually trained on a **4-field
   surface-only** set. This documents the divergence so the numbers are read in
   the right context.

Source of truth: `SEWAA-forecasts-RFE2/SEWAA-forecasts/24h_accumulations/cGAN/`
(`dsrnngan/data.py`, `dsrnngan/config.yaml`, `logs_RFE2_run0{4,5}/eval_validation.txt`).

---

## 1. What actually trained (runs 04 & 05)

| Aspect | Value | Where |
|---|---|---|
| Task | IFS ensemble → RFE2 daily rainfall, **post-processing** (downscaling_factor = 1, same 384×352 grid) | `downscaling_factor.yaml` |
| Predictor fields | **4:** `tp`, `t2m`, `tcwv`, `sp` | `data.py:28` |
| Input channels | **8** = 2 × 4 (ensemble **mean + sd** per field) | `DEFAULT_FCST_SHAPE=(128,128,8)` |
| Static constants | `elev.nc` (orography) + `lsm.nc` (land-sea mask) → 2 ch | `data.py:126` |
| Truth | **RFE2** daily, var `precipitation`, **mm/day → mm/hr (÷24)**, 384×352, pre-regridded onto IFS grid | `data.py:96` |
| Train years | 2018, 2019, 2020, 2021 | `config.yaml` |
| Class sampling weights | `[0.4, 0.3, 0.2, 0.1]` over 4 rain-rate bins | `config.yaml` |
| Rain-rate class bins | `[0.0059, 0.0362, 0.0761]` mm/hr | `tfrecords_generator.py:182` |
| TFRecords | **18 files, ~11 GB**: 16 train (`{year}_1.{0..3}.tfrecords`) + `validation2019/2020.tfrecords` | `/home/ezra/rfe_tfrecords/` |
| num_samples | 204,800 (full schedule, both runs completed) | `config.yaml` |
| Content loss | `ensmeanMSE`, weight 1000 | `config.yaml` |

**run04 vs run05 — only the network capacity differs; data is identical:**

| | run04 | run05 |
|---|---|---|
| `filters_gen` | 32 | **64** |
| `filters_disc` | 128 | **256** |
| everything else | — | identical |

---

## 2. Dataset & variables — runbook **plan** vs runs 04/05 **actual**

The runbook (§0–1) is sized and configured for the **full 14-field** pipeline.
Runs 04/05 used a **trimmed 4-field surface-only** subset (the 14-field line is
present but commented out at `data.py:23`). This is the single most important
context for the CRPS numbers: **the model was never given the full predictor
set the runbook describes.**

| | Runbook plan (§1, §0.1) | Runs 04/05 (actual) |
|---|---|---|
| `all_fcst_fields` | **14:** `cape, cp, mcc, sp, ssr, t2m, tciw, tclw, tcrw, tcw, tcwv, tp, u700, v700` | **4:** `tp, t2m, tcwv, sp` |
| `accumulated_fields` | `cp, ssr, tp` | `tp` |
| `nonnegative_fields` | 13 (all but winds `u700/v700`) | `tp, tcwv` |
| Input channels / patch | **28** (2×14) | **8** (2×4) |
| Raw IFS on prep box | ~**388 GB** (14 fields × 4 yr, ~97 GB/yr) | ~**27 GB/yr** (4 fields; 2020 measured) |
| TFRecords deliverable | ~**14–20 GB** (est.) | ~**11 GB** (measured, 18 files) |
| Vertical / dynamic info | winds `u700/v700`, cloud water `tciw/tclw/tcrw/tcw`, convection `cape/cp`, radiation `ssr` | **none** — surface thermodynamics only |

**What the 4-field set keeps vs drops**

- **Kept:** `tp` (the forecast rainfall itself), `t2m`, `tcwv` (moisture),
  `sp` (surface pressure). Enough for a first bias-correction/calibration pass.
- **Dropped (vs plan):** `cape`, `cp` (convective signals — first-order for EA
  convective rainfall), `u700`/`v700` (low-level jets / moisture transport),
  `tciw/tclw/tcrw/tcw` (cloud & hydrometeor state), `ssr` (radiation/cloud
  proxy), `mcc` (medium cloud cover). These are exactly the predictors the
  variable-selection rationale
  ([docs/east_africa_cgan_variable_selection_rationale.md](docs/east_africa_cgan_variable_selection_rationale.md))
  argues carry the convective/dynamic signal.

> **Implication:** runs 04/05 are a **4-field pilot**, not the runbook's target
> configuration. Adding the dropped convective/dynamic fields is the most
> likely lever for lowering CRPS — see §5.

---

## 3. CRPS results (as evaluated)

Evaluation set: **25 full images × 10 samples** each (`EVAL.num_batches: 25`),
val year 2020. CRPS is in **mm/hr** (truth was converted mm/day → mm/hr).

| Run | Best CRPS | at N (samples) | Final CRPS (N=204,800) | Plateau (last 50 ckpts) |
|---|---|---|---|---|
| **run05** (gen 64 / disc 256) | **0.0839** | 167,936 | 0.0908 | ~0.0887 |
| **run04** (gen 32 / disc 128) | 0.0865 | 157,696 | — | ~0.089 |

- Both **converge by N≈30k** and then oscillate in **~0.084–0.092** with no net
  downward trend — classic GAN behaviour. Training longer is **not** a lever;
  pick the **best checkpoint** (run05 → `gen_weights-0167936.h5`), not the last.
- run05's larger network buys only **0.0865 → 0.0839** — within the sampling
  noise of a 25-image eval.

> ⚠️ **Cross-run comparability caveat:** run01/03 used a *different* validation
> subset (different seed dates) than run04/05, so their CRPS (e.g. run03's
> 0.0809) is **not** directly comparable to run04/05.

---

## 4. Baseline-CRPS comparison (the part that gives 0.084 meaning)

**There is no universal "ideal" CRPS.** CRPS carries the units of the target
(mm/hr) and is bounded below only by the *irreducible* unpredictability of daily
grid-scale rainfall — a perfect score of 0 is unattainable for a genuinely
probabilistic quantity. A CRPS is only interpretable **relative to reference
forecasts on the identical validation set.** The cGAN earns its keep only if it
beats all three baselines below.

### 4.1 Baselines to compute (same 25 images, 2020)

| Baseline | What it is | How to compute CRPS |
|---|---|---|
| **Raw IFS ensemble** | the forecast the cGAN post-processes — the bar it must beat | IFS files store `tp_mean`, `tp_sd` per cell → assume Gaussian ensemble → **closed-form Gaussian CRPS** (§4.2) vs RFE2 truth (÷24 → mm/hr) |
| **Climatology** | RFE2's own distribution, ignoring IFS — the "no-skill" floor | empirical-CDF CRPS of the 2018–2021 RFE2 rainfall distribution vs each 2020 truth field |
| **Quantile mapping (qmap)** | cheap deterministic bias correction of `tp_mean` | fit qmap `tp_mean → RFE2` on train years; deterministic ⇒ CRPS = MAE vs truth |

### 4.2 Closed-form Gaussian CRPS (for the raw-IFS baseline)

For a Gaussian predictive dist. `N(μ, σ)` and observation `y`, with
`z = (y − μ)/σ`, `Φ`/`φ` the standard normal CDF/PDF:

```
CRPS(N(μ,σ), y) = σ · [ z·(2Φ(z) − 1) + 2φ(z) − 1/√π ]
```

Averaging this over all valid (unmasked) cells of the 25 validation images,
using IFS `tp_mean`=μ and `tp_sd`=σ (both ÷24 for mm/hr) against RFE2 truth,
gives the **raw-IFS baseline CRPS** directly comparable to the 0.0839 above.
(IFS precip is non-negative, so this Gaussian assumption slightly over-spreads
near zero — good enough for a baseline; use the ensemble members directly if
higher fidelity is needed.)

### 4.3 Results table (fill in)

| Forecast | CRPS (mm/hr) | Skill vs raw IFS |
|---|---|---|
| Climatology | _TBD_ | — |
| Raw IFS ensemble | _TBD_ | 0 % (reference) |
| Quantile mapping | _TBD_ | _TBD_ |
| **cGAN run04** | **0.0865** | _TBD_ |
| **cGAN run05** | **0.0839** | _TBD_ |

**Reading guide once filled:**
- cGAN **< raw IFS**: post-processing adds value (target: the EA IFS-cGAN
  literature reports ~**10–30 %** CRPS reduction vs raw IFS).
- cGAN **≈ qmap**: the GAN isn't buying much over cheap bias correction —
  revisit predictors/architecture.
- cGAN **≥ climatology**: something is wrong (worse than ignoring the forecast).
- Also inspect **pooled CRPS** (`CRPS_avg_4/16`, already in `eval_validation.txt`)
  and rank histograms for double-penalty / calibration, not just point CRPS.

---

## 5. Verdict & next steps

- **Run is over**; both runs completed the full 204,800-sample schedule and
  have **converged** — more training will not lower CRPS.
- **0.0839 mm/hr is in the expected range** for this domain/units, but is
  **currently un-benchmarked** — compute the §4 baselines before deciding if it
  is good enough. That comparison, not more epochs, is the immediate next step.
- If a real gap remains, the highest-leverage change is **restoring predictors**
  toward the runbook's 14-field plan — especially the dropped **convective and
  dynamic** fields (`cape`, `cp`, `u700`, `v700`) — followed by a **larger /
  more representative evaluation set** (25 images gives noisy CRPS) and trying
  the `CRPS` content loss instead of `ensmeanMSE`.

---

### Provenance
- CRPS: `logs_RFE2_run0{4,5}/eval_validation.txt` (eval finished 2026-07-02).
- Config: `dsrnngan/config.yaml`, `logs_RFE2_run0{4,5}/setup_params.yaml`.
- Fields/truth: `dsrnngan/data.py`, `dsrnngan/tfrecords_generator.py`.
- TFRecords: `/home/ezra/rfe_tfrecords/` (18 files, ~11 GB).
