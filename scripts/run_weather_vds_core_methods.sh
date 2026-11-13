#!/usr/bin/env bash
# DANN and Deep CORAL, one complete run for each weather; no real training in setup.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $SEED" >&2; exit 1; }
PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
WEIGHTS="${WEATHER_VDS_WEIGHTS:-$HERE/pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth}"
EPOCHS="${WEATHER_VDS_EPOCHS:-}"
PROFILE="${WEATHER_VDS_PROFILE:-baseline}"
case "$PROFILE" in baseline|timing_v2|bn_stable) ;; *) echo "Unknown weather profile: $PROFILE" >&2; exit 1 ;; esac
BATCH="${WEATHER_VDS_BATCH_SIZE:-16}"
if [ -n "$EPOCHS" ]; then
  [[ "$EPOCHS" =~ ^[0-9]+$ ]] && [ "$EPOCHS" -ge 1 ] || { echo "Invalid epochs: $EPOCHS" >&2; exit 1; }
fi
[[ "$BATCH" =~ ^[0-9]+$ ]] && [ "$BATCH" -ge 2 ] || { echo "Invalid per-domain batch: $BATCH" >&2; exit 1; }
read -r -a TARGETS <<< "${WEATHER_VDS_TARGETS:-backlight rainy foggy dark}"
for target in "${TARGETS[@]}"; do
  case "$target" in backlight|rainy|foggy|dark) ;; *) echo "Unknown weather: $target" >&2; exit 1 ;; esac
done
export JMDA_DATA_ROOT="$DATA_ROOT"
export PYTHONUNBUFFERED=1
export JMDA_DETERMINISTIC="${JMDA_DETERMINISTIC:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME="${TORCH_HOME:-$HERE/pretrained_weights/torch}"
RUN_DIR="${WEATHER_VDS_OUTPUT_ROOT:-$HERE/runs/weather_vds_core}/weather_uda_${PROFILE}_seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR/logs"
echo "Run directory: $RUN_DIR"
echo "Shared VDS2Raw DANN/Deep CORAL training core; two pretrained ResNet18 encoders; VIS+IR only"
echo "Source=sunny; targets=${TARGETS[*]}; seed=$SEED; profile=$PROFILE; epochs=${EPOCHS:-profile defaults}; batch/domain=$BATCH"
echo "Target training is label-free; complete source/target coverage; peak target_val reported; no model files"
if [ ! -f "$WEIGHTS" ]; then
  "$PYTHON_BIN" -B -u "$HERE/tools/download_official_uda_weights.py" --only resnet18
fi
[ -f "$WEIGHTS" ] || { echo "Missing weights: $WEIGHTS" >&2; exit 1; }
LISTS="$RUN_DIR/prepared_exact_pairs"
"$PYTHON_BIN" -B -u "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT" --output-dir "$LISTS" --targets "${TARGETS[@]}" --modality paired
GPU_ARGS=(--gpus "${JMDA_GPU_CANDIDATES:-0,1}" --min-free-mib "${WEATHER_VDS_MIN_FREE_MIB:-8192}" \
  --max-util-percent "${JMDA_GPU_MAX_UTIL_PERCENT:-10}" --stable-checks "${JMDA_GPU_STABLE_CHECKS:-3}" \
  --poll-seconds "${JMDA_GPU_POLL_SECONDS:-30}")
FAILED=0
EXTRA=()
if [ "${ADAPTATION_MODALITY_DIAGNOSTICS:-0}" = 1 ]; then EXTRA+=(--modality-diagnostics); fi
for target in "${TARGETS[@]}"; do
  for method in dann deepcoral; do
    TAG="${method}_${target}_seed${SEED}"
    LOG="$RUN_DIR/logs/$TAG.log"
    EPOCH_ARGS=()
    if [ -n "$EPOCHS" ]; then EPOCH_ARGS=(--epochs "$EPOCHS"); fi
    echo ">>> $TAG; GPU queue and live training: $LOG"
    if "$PYTHON_BIN" -B -u "$HERE/tools/run_on_available_gpu.py" "${GPU_ARGS[@]}" --tag "$TAG" -- \
      "$PYTHON_BIN" -B -u "$HERE/weather/train_vds_core.py" --method "$method" --target "$target" \
      --data-dir "$LISTS" --save-dir "$RUN_DIR" --weights "$WEIGHTS" \
      --weather-profile "$PROFILE" "${EPOCH_ARGS[@]}" --batch-size "$BATCH" --seed "$SEED" \
      --num-workers "${WEATHER_VDS_NUM_WORKERS:-2}" --eval-batch-size "${WEATHER_VDS_EVAL_BATCH_SIZE:-32}" "${EXTRA[@]}" > "$LOG" 2>&1; then
      echo "    completed: $TAG"
      "$PYTHON_BIN" -B "$HERE/weather/summarize_vds_core.py" --results-dir "$RUN_DIR"
    else
      FAILED=$((FAILED+1))
      echo "    FAILED: $TAG; see $LOG" >&2
      tail -n 15 "$LOG" >&2 || true
    fi
  done
done
echo "Finished: failures=$FAILED; summary=$RUN_DIR/summary; diagnostics=$RUN_DIR/<method>_<target>_seed${SEED}"
[ "$FAILED" -eq 0 ]
