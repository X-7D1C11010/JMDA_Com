#!/usr/bin/env bash
# Invoked from the isolated update directory by its background launcher.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${JMDA_PROJECT_ROOT:?Start with scripts/run_server_mcc_weather_v2_background.sh}"
SEED="${1:-42}"
SCOPE="${2:-all}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $SEED" >&2; exit 1; }
case "$SCOPE" in all|cdan-only) ;; *) echo "usage: $0 SEED {all|cdan-only}" >&2; exit 1 ;; esac
export VDS2RAW_DATA_ROOT="${VDS2RAW_DATA_ROOT:-/home/lixiang/lx/VDS2Raw/threeclass_ready}"
export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
export VDS2RAW_DA_OUTPUT_ROOT="${VDS2RAW_DA_OUTPUT_ROOT:-$PROJECT_ROOT/runs/vds2raw_da}"
export WEATHER_VDS_OUTPUT_ROOT="${WEATHER_VDS_OUTPUT_ROOT:-$PROJECT_ROOT/runs/weather_vds_core}"
export VDS2RAW_DA_WEIGHTS="$PROJECT_ROOT/pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"
export WEATHER_VDS_WEIGHTS="$VDS2RAW_DA_WEIGHTS"
export TORCH_HOME="$PROJECT_ROOT/pretrained_weights/torch"
if [ ! -f "$VDS2RAW_DA_WEIGHTS" ]; then
  "${PYTHON:-python}" -B -u "$PROJECT_ROOT/tools/download_official_uda_weights.py" --only resnet18
fi
export JMDA_DETERMINISTIC=1
export VDS2RAW_DA_BATCH_SIZE=16
export VDS2RAW_NUM_WORKERS=2
unset VDS2RAW_DA_EPOCHS
FAILED=0
if [ "$SCOPE" = all ]; then
  export VDS2RAW_DA_METHODS=mcc
  export VDS2RAW_DA_PROFILE=mcc_v2
  export VDS2RAW_SEEDS="$SEED"
  echo "[1/3] VDS2Raw MCC v2 only, seed=$SEED."
  if ! bash "$HERE/scripts/run_vds2raw_da_methods.sh" formal; then FAILED=$((FAILED+1)); fi
fi

export VDS2RAW_DA_METHODS=cdan
export VDS2RAW_DA_PROFILE=optimized
export VDS2RAW_SEEDS="42 43 44 45 46"
echo "[CDAN] VDS2Raw optimized CDAN, seeds=42 43 44 45 46; validation-CE LR reduction and early stopping."
if ! bash "$HERE/scripts/run_vds2raw_da_methods.sh" formal; then FAILED=$((FAILED+1)); fi

if [ "$SCOPE" = all ]; then
  export WEATHER_VDS_PROFILE=timing_v2
  export WEATHER_VDS_BATCH_SIZE=16
  export WEATHER_VDS_NUM_WORKERS=2
  export WEATHER_VDS_TARGETS="backlight rainy foggy dark"
  unset WEATHER_VDS_EPOCHS
  echo "[3/3] Weather DANN/Deep CORAL timing_v2, four targets, seed=$SEED."
  if ! bash "$HERE/scripts/run_weather_vds_core_methods.sh" "$SEED"; then FAILED=$((FAILED+1)); fi
  echo "Pending experiments finished: failed groups=$FAILED (1 VDS MCC + 5 VDS CDAN + 8 weather tasks requested)."
else
  echo "CDAN append finished: failed groups=$FAILED (5 VDS CDAN tasks requested)."
fi
[ "$FAILED" -eq 0 ]
