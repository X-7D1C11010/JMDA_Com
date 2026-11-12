#!/usr/bin/env bash
# Separate from the existing MCFT/TIR launcher; all five methods use SDA here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;; esac
PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${VDS2RAW_DATA_ROOT:-/home/lixiang/lx/VDS2Raw/threeclass_ready}"
OUTPUT_ROOT="${VDS2RAW_DA_OUTPUT_ROOT:-$HERE/runs/vds2raw_da}"
WEIGHTS="${VDS2RAW_DA_WEIGHTS:-$HERE/pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth}"
EPOCHS="${VDS2RAW_DA_EPOCHS:-100}"
BATCH="${VDS2RAW_DA_BATCH_SIZE:-16}"
METHODS_TEXT="${VDS2RAW_DA_METHODS:-dann deepcoral cdan mdd mcc}"
SEEDS_TEXT="${VDS2RAW_SEEDS:-42}"
read -r -a METHODS <<< "${METHODS_TEXT//,/ }"
read -r -a SEEDS <<< "${SEEDS_TEXT//,/ }"
if [ "${#METHODS[@]}" -eq 0 ] || [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "Method and seed lists must not be empty." >&2; exit 1
fi
[[ "$EPOCHS" =~ ^[0-9]+$ ]] && [ "$EPOCHS" -ge 1 ] || { echo "Invalid epochs: $EPOCHS" >&2; exit 1; }
[[ "$BATCH" =~ ^[0-9]+$ ]] && [ "$BATCH" -ge 2 ] || { echo "Invalid per-domain batch: $BATCH" >&2; exit 1; }
declare -A METHOD_SEEN=()
for method in "${METHODS[@]}"; do
  case "$method" in dann|deepcoral|cdan|mdd|mcc) ;; *) echo "Unknown method: $method" >&2; exit 1 ;; esac
  if [ -n "${METHOD_SEEN[$method]:-}" ]; then echo "Duplicate method: $method" >&2; exit 1; fi
  METHOD_SEEN[$method]=1
done
for seed in "${SEEDS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 1; }
done
export PYTHONUNBUFFERED=1
export JMDA_DETERMINISTIC="${JMDA_DETERMINISTIC:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME="${TORCH_HOME:-$HERE/pretrained_weights/torch}"
EXTRA=()
if [ "$MODE" = smoke ]; then
  EPOCHS=2
  EXTRA=(--max-train-batches 1 --adaptation-warmup-epochs 0 --adaptation-ramp-epochs 1)
fi
if [ ! -f "$WEIGHTS" ]; then
  "$PYTHON_BIN" -B -u "$HERE/tools/download_official_uda_weights.py" --only resnet18
fi
if [ ! -f "$WEIGHTS" ]; then echo "Missing weights: $WEIGHTS" >&2; exit 1; fi
RUN_DIR="$OUTPUT_ROOT/vds2raw_sda_${MODE}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR/logs"
echo "Run directory: $RUN_DIR"
echo "Protocol: labelled source_train + target_train; target_val selection; target_test ONCE after selection"
echo "Inputs: RGB + NIR + full 13 AIS features + 13 validity values; native 96x96"
echo "Methods: ${METHODS[*]}; seeds: ${SEEDS[*]}; epochs=$EPOCHS; batch/domain=$BATCH"
"$PYTHON_BIN" -B -u "$HERE/vds2raw/train_adaptation.py" --audit-only \
  --data-root "$DATA_ROOT" --save-dir "$RUN_DIR" --stats-file "$RUN_DIR/normalization_stats.json"
GPU_ARGS=(--min-free-mib "${VDS2RAW_DA_MIN_FREE_MIB:-6144}" \
  --max-util-percent "${VDS2RAW_GPU_MAX_UTIL_PERCENT:-10}" \
  --stable-checks "${VDS2RAW_GPU_STABLE_CHECKS:-3}" \
  --poll-seconds "${VDS2RAW_GPU_POLL_SECONDS:-30}")
CANDIDATES="${VDS2RAW_GPU_CANDIDATES:-${JMDA_GPU_CANDIDATES:-0,1}}"
GPU_ARGS+=(--gpus "$CANDIDATES")
FAILED=0
for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    TAG="${method}_seed${seed}"
    LOG="$RUN_DIR/logs/$TAG.log"
    echo ">>> $TAG; GPU selection/status and live training: $LOG"
    if "$PYTHON_BIN" -B -u "$HERE/tools/run_on_available_gpu.py" "${GPU_ARGS[@]}" --tag "$TAG" -- \
        "$PYTHON_BIN" -B -u "$HERE/vds2raw/train_adaptation.py" \
        --method "$method" --data-root "$DATA_ROOT" --save-dir "$RUN_DIR" \
        --stats-file "$RUN_DIR/normalization_stats.json" --weights "$WEIGHTS" \
        --epochs "$EPOCHS" --batch-size "$BATCH" --seed "$seed" \
        --num-workers "${VDS2RAW_NUM_WORKERS:-2}" --skip-audit "${EXTRA[@]}" > "$LOG" 2>&1; then
      echo "    completed: $TAG"
      "$PYTHON_BIN" -B "$HERE/vds2raw/summarize_adaptation.py" --results-dir "$RUN_DIR"
    else
      FAILED=$((FAILED+1))
      echo "    FAILED: $TAG; see $LOG" >&2
      tail -n 15 "$LOG" >&2 || true
    fi
  done
done
echo "Finished: failures=$FAILED; results=$RUN_DIR/summary; diagnostics=$RUN_DIR/<method>_seed<seed>"
[ "$FAILED" -eq 0 ]
