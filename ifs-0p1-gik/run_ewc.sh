#!/bin/bash
# The daily workflow on the EWC cluster, for one or more cycles (YYYYMMDDHH; a
# bare YYYYMMDD means 00z):
#   [step1+2 append each cycle to the virtual store and publish it, APPEND_VD=1]
#   step3 Frisky workers -> per cycle: step4 realize to Ceph, step5 check
#   -> step3 stop workers
#   -> per cycle: publish the realized store to source.coop, step5 check the
#      public copy, step6 cGAN NetCDF on the TF frame read from source.coop,
#      step7 cGAN inference [UPLOAD_GAN=1: GAN output -> source.coop],
#      then the IFS NetCDF is deleted (KEEP_NC=1 keeps it)
#
#   CEPH_ENV=... SC_ENV=... MODEL_FOLDER=... CHECKPOINT=316800 bash run_ewc.sh 20261027
#   CEPH_ENV=... SC_ENV=... bash run_ewc.sh 2026102018 2026102100 2026102200
#   SKIP_REALIZE=1 SC_ENV=... bash run_ewc.sh 20261021     # NetCDF + inference only
#
# Env:
#   MODEL_FOLDER, CHECKPOINT  the trained TF cGAN (setup_params.yaml, models/);
#                             unset -> step 7 runs --dry-run only (inputs checked)
#   CONSTANTS                 elev.nc lsm.nc FCSTNorm2018.pkl (default: the copy in
#                             ../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS)
#   TFPY                      python with TensorFlow for step 7 (default $PY)
#   UPLOAD_GAN=1              upload GAN_IFS_*.nc to source.coop (config.GAN_SOURCECOOP_ROOT)
#   KEEP_NC=1                 keep IFS_*.nc in OUTDIR (default: deleted after step 7)
#   SKIP_REALIZE=1            cycles are already realized and published
#
# One Icechunk store per cycle, named <YYYYMMDD>T<HH>z, so dropping a cycle is
# a prefix delete. 06z/18z runs stop at 144 h: their stores and NetCDFs hold
# the 25 lead times 0..144 h instead of 29.
#
# Never runs while the CRMA daily routine is active: it waits for it, because
# both use every worker VM. The workers here are separate processes
# (/tmp/frisky-cgan-ea, :8798) and are stopped again before publishing.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$D/.venv/bin/python}
TFPY=${TFPY:-$PY}
DASKPY=${DASKPY:-/opt/mamba/envs/dask/bin/python}   # has `distributed`, for step3
[ $# -ge 1 ] || { echo "usage: run_ewc.sh YYYYMMDD[HH] ..."; exit 2; }
SCHED_IP=${SCHED_IP:-192.168.1.164}
cfg() { $PY -c "import sys; sys.path.insert(0, '$D'); import config; print(config.$1)"; }
TAG=$(cfg BOX_TAG)
CEPH_ROOT=cgan14-$TAG
SC_ROOT=forecasts/ecmwf-ifs-0p1-cgan14-$TAG
GAN_ROOT=$(cfg GAN_SOURCECOOP_ROOT)
CONSTANTS=${CONSTANTS:-$D/../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS}
OUT=${OUTDIR:-$D/nc}
GAN_OUT=${GAN_OUTDIR:-$D/gan}
: "${SC_ENV:?set SC_ENV to the source.coop .env}"
[ "${SKIP_REALIZE:-0}" = 1 ] || : "${CEPH_ENV:?set CEPH_ENV to the Ceph AK/SK file}"
NOAWS="env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN -u AWS_ENDPOINT_URL"
F='WARN|\.rs:|^\s*$'
mkdir -p "$D/logs"
CYCLES=()
for c in "$@"; do [ ${#c} = 8 ] && c=${c}00; CYCLES+=($c); done
name() { echo "${1:0:8}T${1:8:2}z"; }
sc() {  # run a command with the source.coop STS credentials, in a subshell only
  ( set -a; . "$SC_ENV"; set +a; eval "$(source-coop creds --format env)"
    AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt "$@" ); }

GOOD=()
if [ "${SKIP_REALIZE:-0}" = 1 ]; then
  GOOD=("${CYCLES[@]}")
else
  while pgrep -f 'run_2425.sh|3-realize-frisky/daily_run.sh' >/dev/null; do
    echo "=== CRMA daily run active, waiting $(date -u +%FT%TZ)"; sleep 300; done

  if [ "${APPEND_VD:-0}" = 1 ]; then
    echo "=== step1+2 append ${CYCLES[*]} to the virtual store, publish"
    PUBLISH=1 SC_ENV=$SC_ENV bash "$D/step1_append_when_published.sh" "${CYCLES[@]}"
  fi

  echo "=== step3 workers"
  $DASKPY "$D/step3_deploy_workers.py" --install 2>&1 | grep -E 'ok=|FAILED|ready' | tail -2
  $DASKPY "$D/step3_deploy_workers.py" --scheduler-on $SCHED_IP --start --workers-per-vm 1 \
      --nthreads ${NTHREADS:-6} --memory-limit 12GB 2>&1 | grep -E 'scheduler |frisky workers'
  for c in "${CYCLES[@]}"; do
    n=$(name $c)
    $DASKPY "$D/mem_monitor.py" --every 15 > "$D/logs/mem-$n.tsv" 2>/dev/null & MON=$!
    echo "=== $n step4 realize -> ceph:$CEPH_ROOT/$n  $(date -u +%FT%TZ)"
    $PY -u "$D/step4_realize.py" --date ${c:0:8} --run ${c:8:2} --sink "ceph:$CEPH_ROOT/$n" \
        --env "$CEPH_ENV" --scheduler $SCHED_IP:8798 2>&1 | grep -v -E "$F"
    rc=${PIPESTATUS[0]}
    kill $MON
    [ $rc = 0 ] || { echo "=== $n realize failed; not published"; continue; }
    echo "=== $n step5 check (Ceph)"
    $PY -u "$D/step5_check.py" --store "ceph:$CEPH_ROOT/$n" --env "$CEPH_ENV" 2>&1 \
        | grep -v -E "$F" | grep -E 'complete|bit-exact|PROBLEMS|VERDICT'
    [ ${PIPESTATUS[0]} = 0 ] || { echo "=== $n check failed; not published"; continue; }
    GOOD+=($c)
  done
  echo "=== step3 stop workers"
  $DASKPY "$D/step3_deploy_workers.py" --stop 2>&1 | grep -c "still_running': '0'"

  # publish every cycle first: it takes ~30 s each, the NetCDF + inference minutes
  for c in "${GOOD[@]}"; do
    n=$(name $c)
    echo "=== $n publish -> source.coop $SC_ROOT/$n"
    sc $PY -u "$D/publish_ceph_to_source_coop.py" --src-prefix $CEPH_ROOT/$n \
        --dest $SC_ROOT/$n --env "$CEPH_ENV" 2>&1 | grep -vE '^\s+retry|WARN' | tail -2
    echo "=== $n step5 check (source.coop, anonymous)"
    $NOAWS $PY -u "$D/step5_check.py" --store "sourcecoop:$SC_ROOT/$n" --samples 4 2>&1 \
        | grep -v -E "$F" | grep -E 'complete|bit-exact|VERDICT'
  done
fi

for c in "${GOOD[@]}"; do
  n=$(name $c); f=IFS_${c:0:8}_${c:8:2}Z.nc
  echo "=== $n step6 cGAN NetCDF (TF frame) from source.coop  $(date -u +%FT%TZ)"
  $NOAWS $PY -u "$D/step6_make_cgan_netcdf.py" --store "sourcecoop:$SC_ROOT/$n" --outdir "$OUT" \
      --frame tf --members ${NC_MEMBERS:-perturbed} 2>&1 | grep -v -E "$F" \
      | grep -E 'WARNING|wrote|non-finite|widen'
  [ -f "$OUT/$f" ] || { echo "=== $n no NetCDF"; continue; }

  if [ -n "${MODEL_FOLDER:-}" ] && [ -n "${CHECKPOINT:-}" ]; then
    echo "=== $n step7 cGAN inference"
    $TFPY -u "$D/step7_infer.py" --nc "$OUT/$f" --constants "$CONSTANTS" \
        --model-folder "$MODEL_FOLDER" --checkpoint $CHECKPOINT --outdir "$GAN_OUT" \
        --ensemble-members ${GAN_MEMBERS:-50} 2>&1 | grep -E 'model|wrote|Error|error' | tail -3
    if [ "${UPLOAD_GAN:-0}" = 1 ] && [ -f "$GAN_OUT/GAN_$f" ]; then
      echo "=== $n upload GAN output -> source.coop $GAN_ROOT/$n"
      sc $PY "$D/upload_files.py" --to "sourcecoop:$GAN_ROOT/$n" "$GAN_OUT/GAN_$f"
    fi
  else
    echo "=== $n step7 dry run (MODEL_FOLDER/CHECKPOINT not set: inputs checked, no inference)"
    $PY -u "$D/step7_infer.py" --nc "$OUT/$f" --constants "$CONSTANTS" --dry-run 2>&1 \
        | grep -E 'input|dry run|non-finite|frame'
  fi
  [ "${KEEP_NC:-0}" = 1 ] || { rm -f "$OUT/$f"; echo "    removed $OUT/$f"; }
done
echo "=== done ${GOOD[*]} of ${CYCLES[*]}  $(date -u +%FT%TZ)"
