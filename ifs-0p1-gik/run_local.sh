#!/bin/bash
# The whole workflow on ONE machine, with no source.coop and no Ceph:
#   step1 virtual store (local dir) -> step4 realize (local Frisky cluster,
#   local dir) -> step5 check -> step6 cGAN NetCDF
#
#   bash run_local.sh 20261021            # 00z, all 14 fields x 51 x 29
#   STEPS="0 6 12" MEMBERS=5 bash run_local.sh 20261021     # a small trial
#
# Env: IFS0P1_LOCAL_ROOT (default ~/icechunk-local), WORKERS (2), THREADS (4),
#      OUTDIR (./nc), NC_MEMBERS (perturbed)
# Disk: ~13 MB per cycle of virtual store, ~2.4 GB per realized cycle,
#       ~3-5 GB per full NetCDF.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$D/.venv/bin/python}
DATE=${1:?YYYYMMDD}; RUN=${RUN:-00}
ROOT=${IFS0P1_LOCAL_ROOT:-$HOME/icechunk-local}
VD=$ROOT/ecmwf-ifs-0p1-icechunk-vd
TAG=$($PY -c "import sys; sys.path.insert(0, '$D'); import config; print(config.BOX_TAG)")
SINK=$ROOT/ecmwf-ifs-0p1-cgan14-$TAG/${DATE}T${RUN}z
EXTRA=()
[ -n "${STEPS:-}" ] && EXTRA+=(--steps $STEPS)
[ -n "${MEMBERS:-}" ] && EXTRA+=(--members $MEMBERS)
mkdir -p "$D/logs" "$ROOT"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_ENDPOINT_URL || true

echo "=== step1 virtual store $VD"
if ! $PY - "$VD" "$DATE" "$RUN" <<'EOF'
import sys, numpy as np, icechunk as ic, zarr
vd, d, r = sys.argv[1:]
st = ic.local_filesystem_storage(vd)
if not ic.Repository.exists(st):
    sys.exit(1)
g = zarr.open_group(ic.Repository.open(st).readonly_session("main").store,
                    path=f"0p1/{r}z", mode="r")
want = (np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}T{r}") - np.datetime64("1970-01-01T00")) // np.timedelta64(1, "h")
sys.exit(0 if want in g["time"][:] else 1)
EOF
then
  $PY "$D/step1_build_virtual.py" --date $DATE --run $RUN --resolution 0p1 --store "$VD" --verify 8 \
      2>&1 | grep -v -E 'WARN|\.rs:|^\s*$'
else
  echo "    ${DATE} ${RUN}z already in the local virtual store"
fi

echo "=== step4 realize -> $SINK"
$PY -u "$D/step4_realize.py" --date $DATE --run $RUN --vd "local:$VD" --sink "local:$SINK" \
    --workers ${WORKERS:-2} --threads ${THREADS:-4} "${EXTRA[@]}" 2>&1 | grep -v -E 'WARN|\.rs:|^\s*$'

echo "=== step5 check"
$PY -u "$D/step5_check.py" --store "local:$SINK" --vd "local:$VD" --samples 8 \
    2>&1 | grep -v -E 'WARN|\.rs:|^\s*$'

echo "=== step6 cGAN NetCDF"
$PY -u "$D/step6_make_cgan_netcdf.py" --store "local:$SINK" --outdir "${OUTDIR:-$D/nc}" \
    --members ${NC_MEMBERS:-perturbed} 2>&1 | grep -v -E 'WARN|\.rs:'
