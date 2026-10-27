#!/bin/bash
# The whole workflow on the EWC cluster, published to source.coop:
#   [step1+2 append the cycle to the virtual store and publish it, APPEND_VD=1]
#   step3 Frisky workers -> step4 realize to Ceph -> step5 check -> step3 stop
#   -> publish the realized cycle to source.coop -> step5 check the public copy
#   -> step6 cGAN NetCDF read from source.coop
#
#   CEPH_ENV=~/.env.ceph SC_ENV=~/.env.sourcecoop bash run_ewc.sh 20261021
#
# Never runs while the CRMA daily routine is active: it waits for it, because
# both use every worker VM. The workers here are separate processes
# (/tmp/frisky-cgan-ea, :8798) and are stopped again before publishing.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$D/.venv/bin/python}
DASKPY=${DASKPY:-/opt/mamba/envs/dask/bin/python}   # has `distributed`, for step3
DATE=${1:?YYYYMMDD}; RUN=${RUN:-00}
SCHED_IP=${SCHED_IP:-192.168.1.164}
CEPH=cgan14-20e53e-25n15s/${DATE}${RUN}z
SC=forecasts/ecmwf-ifs-0p1-cgan14-20e53e-25n15s/${DATE}${RUN}z
: "${CEPH_ENV:?set CEPH_ENV to the Ceph AK/SK file}" "${SC_ENV:?set SC_ENV to the source.coop .env}"
NOAWS="env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN -u AWS_ENDPOINT_URL"
F='WARN|\.rs:|^\s*$'
mkdir -p "$D/logs"

while pgrep -f 'run_2425.sh|3-realize-frisky/daily_run.sh' >/dev/null; do
  echo "=== CRMA daily run active, waiting $(date -u +%FT%TZ)"; sleep 300; done

if [ "${APPEND_VD:-0}" = 1 ]; then
  echo "=== step1+2 append ${DATE}${RUN} to the virtual store, publish"
  PUBLISH=1 SC_ENV=$SC_ENV bash "$D/step1_append_when_published.sh" ${DATE}${RUN}
fi

echo "=== step3 workers"
$DASKPY "$D/step3_deploy_workers.py" --install 2>&1 | grep -E 'ok=|FAILED|ready' | tail -2
$DASKPY "$D/step3_deploy_workers.py" --scheduler-on $SCHED_IP --start --workers-per-vm 1 \
    --nthreads ${NTHREADS:-6} --memory-limit 12GB 2>&1 | grep -E 'scheduler |frisky workers'
$DASKPY "$D/mem_monitor.py" --every 15 > "$D/logs/mem-${DATE}${RUN}z.tsv" 2>/dev/null & MON=$!

echo "=== step4 realize -> ceph:$CEPH"
$PY -u "$D/step4_realize.py" --date $DATE --run $RUN --sink "ceph:$CEPH" --env "$CEPH_ENV" \
    --scheduler $SCHED_IP:8798 2>&1 | grep -v -E "$F"
rc=${PIPESTATUS[0]}
kill $MON
echo "=== step3 stop workers"
$DASKPY "$D/step3_deploy_workers.py" --stop 2>&1 | grep -c "still_running': '0'"
[ $rc = 0 ] || { echo "=== realize failed; nothing published"; exit 1; }

echo "=== step5 check (Ceph)"
$PY -u "$D/step5_check.py" --store "ceph:$CEPH" --env "$CEPH_ENV" 2>&1 | grep -v -E "$F" \
    | grep -E 'complete|bit-exact|PROBLEMS|VERDICT'
[ ${PIPESTATUS[0]} = 0 ] || { echo "=== check failed; nothing published"; exit 1; }

echo "=== publish -> source.coop $SC"
( set -a; . "$SC_ENV"; set +a; eval "$(source-coop creds --format env)"
  AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt $PY -u "$D/publish_ceph_to_source_coop.py" \
      --src-prefix $CEPH --dest $SC --env "$CEPH_ENV" ) 2>&1 | grep -vE '^\s+retry|WARN' | tail -3

echo "=== step5 check (source.coop, anonymous)"
$NOAWS $PY -u "$D/step5_check.py" --store "sourcecoop:$SC" --samples 4 2>&1 | grep -v -E "$F" \
    | grep -E 'complete|bit-exact|VERDICT'

echo "=== step6 cGAN NetCDF from source.coop"
$NOAWS $PY -u "$D/step6_make_cgan_netcdf.py" --store "sourcecoop:$SC" --outdir "${OUTDIR:-$D/nc}" \
    --members ${NC_MEMBERS:-perturbed} 2>&1 | grep -v -E "$F" | tail -1
