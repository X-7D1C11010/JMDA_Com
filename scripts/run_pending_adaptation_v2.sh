#!/usr/bin/env bash
# Invoked from the isolated update directory by its background launcher.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${JMDA_PROJECT_ROOT:?Start with scripts/run_server_mcc_weather_v2_background.sh}"
SEED="${1:-42}"
SCOPE="${2:-vds-only}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $SEED" >&2; exit 1; }
case "$SCOPE" in all|cdan-only|v3|v4|vds-only) ;; *) echo "usage: $0 SEED {all|cdan-only|v3|v4|vds-only}" >&2; exit 1 ;; esac
if [ "$SCOPE" = v4 ]; then
  echo "Weather balanced_v4 did not pass the local seed42 screen; the combined v4 queue is not released." >&2
  echo "Use scope vds-only for the verified heading/size MCC and CDAN configurations. No training was started." >&2
  exit 2
fi
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
export VDS2RAW_AIS_FIELDS=heading_size
unset VDS2RAW_DA_EPOCHS
FAILED=0
if [ "$SCOPE" = v3 ] || [ "$SCOPE" = vds-only ]; then
  AUDIT_DIR="$PROJECT_ROOT/results/adaptation_${SCOPE}_$(date +%Y%m%d_%H%M%S)_$$"
  "${PYTHON:-python}" -B -u "$HERE/tools/audit_vds2raw_inputs.py" --data-root "$VDS2RAW_DATA_ROOT" --output "$AUDIT_DIR/dataset_input_audit.json" --ais-fields "$VDS2RAW_AIS_FIELDS"
  export ADAPTATION_MODALITY_DIAGNOSTICS=1
  export VDS2RAW_DA_METHODS=mcc
  export VDS2RAW_DA_PROFILE=mcc_v3
  export VDS2RAW_SEEDS="42 43 44 45 46"
  echo "[$SCOPE VDS 1/2] MCC v3 seeds=$VDS2RAW_SEEDS: normalized modality features and post-ramp validation controls."
  if ! bash "$HERE/scripts/run_vds2raw_da_methods.sh" formal; then FAILED=$((FAILED+1)); fi
  export VDS2RAW_DA_METHODS=cdan
  export VDS2RAW_DA_PROFILE=optimized
  if [ "$SCOPE" = vds-only ]; then export VDS2RAW_DA_PROFILE=cdan_safe_v4; fi
  echo "[$SCOPE VDS 2/2] CDAN profile=$VDS2RAW_DA_PROFILE seeds=$VDS2RAW_SEEDS; heading/size AIS only; validation-based model selection."
  if ! bash "$HERE/scripts/run_vds2raw_da_methods.sh" formal; then FAILED=$((FAILED+1)); fi
  if [ "$SCOPE" = vds-only ]; then
    echo "VDS-only finished: failed groups=$FAILED (MCC x5 + CDAN x5, seeds 42-46; no weather tasks)."
    [ "$FAILED" -eq 0 ]
    exit
  fi
  export WEATHER_VDS_BATCH_SIZE=16
  export WEATHER_VDS_NUM_WORKERS=2
  export WEATHER_VDS_TARGETS="backlight rainy foggy dark"
  unset WEATHER_VDS_EPOCHS
  export WEATHER_VDS_METHODS="dann deepcoral"
  for WEATHER_VDS_PROFILE in baseline bn_stable; do
    export WEATHER_VDS_PROFILE
    echo "[v3 weather] $WEATHER_VDS_PROFILE, identical baseline LR/budget/loss settings; BN is the only changed factor."
    if ! bash "$HERE/scripts/run_weather_vds_core_methods.sh" "$SEED"; then FAILED=$((FAILED+1)); fi
  done
  echo "v3 finished: failed groups=$FAILED (MCC x5 + CDAN x5, seeds 42-46; then weather baseline x8 + BN-stable x8, seed $SEED)."
  [ "$FAILED" -eq 0 ]
  exit
fi
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
