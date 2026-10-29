# Provenance of the SEWAA (production) cGAN method used here

The `sewaa-6h` and `sewaa-7d` paths in this folder (`step6_sewaa_netcdf.py`,
`step7_infer.py --product ...`) reproduce what the SEWAA forecast package runs
operationally, but self-contained: no Docker, no fast-cgan job scheduler, no SFTP
pull of a ready-made `IFS_*.nc` from the external IFS server. The input NetCDF is
built here from the ECMWF open data (virtual Icechunk store -> realized store),
and the inference runs from the code vendored in `ensemble_cgan/`.

Everything below was read on 2026-10-29 at these exact versions.

## Source repositories and versions

| Repository | Branch | Commit | Commit date | Licence | Used as |
|---|---|---|---|---|---|
| github.com/icpac-igad/sewaa-forecasts-package | main | `3665aea2ae93af4524e1f753933b2e9dc6cd6915` | 2025-07-15 | GPL-3.0 | which job runs which model, with which `forecast-*.yaml` (cited, not copied) |
| github.com/icpac-igad/fast-cgan | main | `b3cc3b31ebc7adfa9ca353f891953fcd27b97337` | 2025-06-25 | GPL-3.0 | the operational job flow (cited, not copied) |
| github.com/jaysnm/ensemble-cgan | Jurre_brishti | `cb6377dc06a66ed52796974b79662dd18e961238` | 2025-01-26 | MIT | 6 h model code, **vendored** |
| github.com/jaysnm/ensemble-cgan | Mvua_kubwa | `fb0dd0da8b7d3992ee0881688ebd43779b98dfa4` | 2025-02-28 | MIT | 24 h / 7 day model code, **vendored** |

The GPL-3.0 repositories are cited by path and commit only, so that this folder
does not take on their licence. ensemble-cgan is MIT, and its `LICENSE` is kept
next to the copied files.

## Vendored files (`ensemble_cgan/`, byte-identical copies)

| File here | Source path | Note |
|---|---|---|
| `models.py`, `blocks.py`, `layers.py`, `noise.py`, `downscaling_factor.yaml` | `dsrnngan/<same name>` | identical in both branches at the commits above; `step7_infer.py` builds the generator from these |
| `reference/test_forecast.Jurre_brishti.py` | `Jurre_brishti:dsrnngan/test_forecast.py` | the 6 h operational inference; reference only, not run |
| `reference/forecast_date.Mvua_kubwa.py` | `Mvua_kubwa:dsrnngan/forecast_date.py` | the 7 day operational inference; reference only, not run |
| `reference/data.Jurre_brishti.py` | `Jurre_brishti:dsrnngan/data.py` | `load_fcst()` documents the accumulation convention; reference only |
| `LICENSE` | `LICENSE` | MIT, identical in both branches |

## The operational routine (as of the commits above)

`sewaa-forecasts-package/docker-compose.yml` starts one container per job, all
from the `icpac/cgan-forecast-model` image (`jobs.Dockerfile` clones
fast-cgan plus the two ensemble-cgan branches):

| Service | Command | Script run (cwd `<model>/ensemble-cgan/dsrnngan`) | `configs/<model>/forecast-*.yaml` |
|---|---|---|---|
| `jurre-brishti-ens` | `fastcgan/jobs/cgan_ifs.py -m jurre-brishti-ens` | `test_forecast.py -f <IFS file>` | `start_hour: 30`, `end_hour: 54`, checkpoint `316800`, 50 members |
| `jurre-brishti-counts` | `... -m jurre-brishti-count` | same | same, 1000 members |
| `mvua-kubwa-ens` | `... -m mvua-kubwa-ens` | `forecast_date.py -f <IFS file>` | `start_hour: 6`, `end_hour: 174`, checkpoint `300800`, 50 members |
| `mvua-kubwa-counts` | `... -m mvua-kubwa-count` | same | same, 1000 members |

Each hour, `fastcgan/jobs/cgan_ifs.py`:

1. `syncronize_post_processed_ifs_data`: `fastcgan/jobs/sftp.py`
   `sync_sftp_data_files` copies new `IFS_<YYYYMMDD>_<HH>Z.nc` files from
   `$IFS_SERVER_HOST:/data/Operational` (6 h) or `/data/Operational_7d` (7 day).
   If no server is configured, `fastcgan/jobs/icpac_ftp.py` downloads them from
   the ICPAC FTP instead. **The IFS NetCDF is made elsewhere; the package only
   consumes it.**
2. `post_process_downloaded_cgan_ifs` files them under `cgan-ifs-6h-ens` or
   `cgan-ifs-7d-ens`.
3. `generate_cgan_forecasts` runs the model script for each date that has no
   GAN output yet. If the run fails and the IFS file is under 40 MB, the file
   is deleted.

## The input file both scripts read (`IFS_<YYYYMMDD>_<HH>Z.nc`)

Derived from the reference scripts; `step6_sewaa_netcdf.py` writes it.

```
time        (time=1)            hours since 1900-01-01 00:00:00.0
valid_time  (valid_time)        hours since 1900-01-01
latitude    (384) -13.65..24.65 ascending; longitude (352) 19.15..54.25
<f>_ensemble_mean                (valid_time, latitude, longitude)
<f>_ensemble_standard_deviation  (valid_time, latitude, longitude)
   for the 14 fields cape cp mcc sp ssr t2m tciw tclw tcrw tcw tcwv tp u700 v700
```

- **Accumulated fields (tp, cp, ssr) are pre-differenced.** Index j holds the
  increment from valid time j to j+1. `data.py load_fcst()` says so ("data[:, j]
  has accumulation between times j and j+1"). Both scripts take mean/sd at the
  window *start* index for them.
- **6 h (`test_forecast.py`).** Its loop is `range(start_hour // 6 - 5,
  end_hour // 6 - 5)`, which hard-codes that the file **starts at +30 h**. With
  `30..54` it reads indices 0..4: the valid times **+30, 36, 42, 48, 54 h**, giving 4
  windows. Per window: accumulated `[mean_j, sd_j, 0, 0]`, others
  `[mean_j, sd_j, mean_j+1, sd_j+1]`, which is 56 channels.
- **7 day (`forecast_date.py`).** The file starts at +0 h with a 6 h step, and
  it hard-codes `in_time_idx = [1, 5, 9, 13, 17, 21, 25]`. Day d uses indices
  4d+1 .. 4d+5, i.e. **+6 h .. +174 h**, giving 7 days of 24 h each starting at +6, 30, …,
  150 h. Accumulated: mean of the 4 window means, RMS of the 4 sds. Others:
  trapezium rule over the 5 instants, for the means and for the variances. That is
  2 channels per field (28). The output name is hard-coded `GAN_<date>_00Z.nc`.
- **Not stated in either script.** The script that makes these files on the IFS
  server is not in any of the repositories, so two things are assumptions here.
  First, the members: we use all 51 (the control plus 50 perturbed, as in
  `gefs-gik-data-prepration/cgan_ecmwf/ECMWF_cGAN_INFERENCE_WORKFLOW.md`).
  Second, the SD: we use the population SD (`ddof=0`, numpy's default, as in
  `stream_cgan_variables.py`). Both can be changed with `--members` and `--ddof`
  on `step6_sewaa_netcdf.py`. Check them against one operational file when one
  is available.

The weights and constants are not public at a working URL: the URLs in
`manager.sh` (`cgan.icpac.net/ftp/models-config/{jurre-brishti,mvua-kubwa}.tar.gz`)
returned 404 on 2026-10-29. Each model needs its own folder (`setup_params.yaml`,
`models/gen_weights-<checkpoint>.h5`) and its own constants (`elev.nc`, `lsm.nc`,
`FCSTNorm2018.pkl`).
