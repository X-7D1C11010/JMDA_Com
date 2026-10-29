#!/usr/bin/env bash
# Sequential VDS2Raw runner. Override VDS2RAW_METHOD_ORDER when a different
# resource-aware order is needed, for example: "dsne ccsa mcft3".
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}"; exit 1 ;; esac

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${VDS2RAW_DATA_ROOT:-/home/lixiang/lx/VDS2Raw/threeclass_ready}"
OUTPUT_ROOT="${VDS2RAW_OUTPUT_ROOT:-$HERE/runs/vds2raw}"
read -r -a SEEDS <<< "${VDS2RAW_SEEDS:-42}"
read -r -a METHOD_ORDER <<< "${VDS2RAW_METHOD_ORDER:-ccsa dsne mcft3}"
CCSA_EPOCHS="${VDS2RAW_CCSA_EPOCHS:-80}"
DSNE_EPOCHS="${VDS2RAW_DSNE_EPOCHS:-20}"
MCFT_EPOCHS="${VDS2RAW_MCFT_EPOCHS:-500}"
CCSA_SOURCE_EPOCHS="${VDS2RAW_CCSA_SOURCE_EPOCHS:-50}"
DSNE_SOURCE_EPOCHS="${VDS2RAW_DSNE_SOURCE_EPOCHS:-50}"
CCSA_BATCH="${VDS2RAW_CCSA_BATCH_SIZE:-256}"
DSNE_BATCH="${VDS2RAW_DSNE_BATCH_SIZE:-32}"
MCFT_BATCH="${VDS2RAW_MCFT_BATCH_SIZE:-32}"
DSNE_LAYERS="${VDS2RAW_DSNE_LAYERS:-101}"
LEGACY_TARGET_PER_CLASS="${VDS2RAW_TARGET_SAMPLES_PER_CLASS:-}"
CCSA_TARGET_PER_CLASS="${VDS2RAW_CCSA_TARGET_SAMPLES_PER_CLASS:-${LEGACY_TARGET_PER_CLASS:-10}}"
DSNE_TARGET_PER_CLASS="${VDS2RAW_DSNE_TARGET_SAMPLES_PER_CLASS:-${LEGACY_TARGET_PER_CLASS:-0}}"
MCFT_EVAL="${VDS2RAW_MCFT_EVAL_FREQUENCY:-10}"
GPU_MAX_UTIL="${VDS2RAW_GPU_MAX_UTIL_PERCENT:-10}"
GPU_STABLE_CHECKS="${VDS2RAW_GPU_STABLE_CHECKS:-3}"
GPU_POLL_SECONDS="${VDS2RAW_GPU_POLL_SECONDS:-30}"
GPU_CANDIDATES="${VDS2RAW_GPU_CANDIDATES:-}"
CCSA_MIN_FREE_MIB="${VDS2RAW_CCSA_MIN_FREE_MIB:-18000}"
DSNE_MIN_FREE_MIB="${VDS2RAW_DSNE_MIN_FREE_MIB:-30000}"
MCFT_MIN_FREE_MIB="${VDS2RAW_MCFT_MIN_FREE_MIB:-10000}"
if [ "${#METHOD_ORDER[@]}" -ne 3 ]; then
  echo "VDS2RAW_METHOD_ORDER must contain ccsa, dsne and mcft3 exactly once." >&2
  exit 1
fi
CCSA_ORDER_COUNT=0; DSNE_ORDER_COUNT=0; MCFT_ORDER_COUNT=0
for method in "${METHOD_ORDER[@]}"; do
  case "$method" in
    ccsa) CCSA_ORDER_COUNT=$((CCSA_ORDER_COUNT + 1)) ;;
    dsne) DSNE_ORDER_COUNT=$((DSNE_ORDER_COUNT + 1)) ;;
    mcft3) MCFT_ORDER_COUNT=$((MCFT_ORDER_COUNT + 1)) ;;
    *) echo "Unknown VDS2Raw method in order: $method" >&2; exit 1 ;;
  esac
done
if [ "$CCSA_ORDER_COUNT" -ne 1 ] || [ "$DSNE_ORDER_COUNT" -ne 1 ] \
  || [ "$MCFT_ORDER_COUNT" -ne 1 ]; then
  echo "VDS2RAW_METHOD_ORDER must contain ccsa, dsne and mcft3 exactly once." >&2
  exit 1
fi
if [ -n "${VDS2RAW_CCSA_SOURCE_CHECKPOINT:-}" ] && \
   [ -n "${VDS2RAW_CCSA_PRETRAINED_BACKBONE:-}" ]; then
  echo "Set only one CCSA source checkpoint or pretrained backbone." >&2
  exit 1
fi
if [ -n "${VDS2RAW_DSNE_SOURCE_CHECKPOINT:-}" ] && \
   [ -n "${VDS2RAW_DSNE_PRETRAINED_BACKBONE:-}" ]; then
  echo "Set only one d-SNE source checkpoint or pretrained backbone." >&2
  exit 1
fi
MAX_BATCH_ARGS=()
PRETRAINED_ARGS=()
if [ "$MODE" = smoke ]; then
  CCSA_EPOCHS=1; DSNE_EPOCHS=1; MCFT_EPOCHS=1
  CCSA_SOURCE_EPOCHS=1; DSNE_SOURCE_EPOCHS=1
  CCSA_BATCH=8; DSNE_BATCH=4; MCFT_BATCH=4; DSNE_LAYERS=101; MCFT_EVAL=1
  MAX_BATCH_ARGS=(--max-train-batches 1)
  PRETRAINED_ARGS=(--no-pretrained)
fi

RUN_ID="vds2raw_${MODE}_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
ARTIFACT_DIR="$RUN_DIR/artifacts"
LOG_DIR="$RUN_DIR/logs"
SUMMARY_DIR="$RUN_DIR/summary"
STATS_FILE="$RUN_DIR/normalization_stats.json"
mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$SUMMARY_DIR"

echo "Run directory: $RUN_DIR"
echo "Method order: ${METHOD_ORDER[*]}"
"$PYTHON_BIN" "$HERE/vds2raw/audit.py" --data-root "$DATA_ROOT" \
  --stats-file "$STATS_FILE" --report-file "$RUN_DIR/dataset_audit.json"

run_one() {
  local tag="$1" min_free_mib="$2"; shift 2
  local log="$LOG_DIR/$tag.log"
  local gpu_args=(--min-free-mib "$min_free_mib" \
    --max-util-percent "$GPU_MAX_UTIL" \
    --stable-checks "$GPU_STABLE_CHECKS" \
    --poll-seconds "$GPU_POLL_SECONDS" --tag "$tag")
  if [ -n "$GPU_CANDIDATES" ]; then
    gpu_args+=(--gpus "$GPU_CANDIDATES")
  fi
  echo ">>> $tag (GPU gate: free>=${min_free_mib} MiB, util<=${GPU_MAX_UTIL}%; log=$log)"
  if "$PYTHON_BIN" "$HERE/tools/run_on_available_gpu.py" \
    "${gpu_args[@]}" -- "$@" > "$log" 2>&1; then
    echo "    completed; log: $log"
  else
    tail -n 40 "$log" || true
    echo "FAILED: $tag; log: $log"
    exit 1
  fi
}

for seed in "${SEEDS[@]}"; do
  COMMON=(--data-root "$DATA_ROOT" --save-dir "$ARTIFACT_DIR" \
    --stats-file "$STATS_FILE" --seed "$seed" --num-workers "${VDS2RAW_NUM_WORKERS:-2}" --skip-audit)
  CCSA_CKPT_ARGS=()
  DSNE_CKPT_ARGS=()
  CCSA_BACKBONE_ARGS=()
  DSNE_BACKBONE_ARGS=()
  if [ -n "${VDS2RAW_CCSA_SOURCE_CHECKPOINT:-}" ]; then
    CCSA_CKPT_ARGS=(--source-checkpoint "$VDS2RAW_CCSA_SOURCE_CHECKPOINT")
  fi
  if [ -n "${VDS2RAW_DSNE_SOURCE_CHECKPOINT:-}" ]; then
    DSNE_CKPT_ARGS=(--source-checkpoint "$VDS2RAW_DSNE_SOURCE_CHECKPOINT")
  fi
  if [ -n "${VDS2RAW_CCSA_PRETRAINED_BACKBONE:-}" ]; then
    CCSA_BACKBONE_ARGS=(--pretrained-backbone "$VDS2RAW_CCSA_PRETRAINED_BACKBONE")
  fi
  if [ -n "${VDS2RAW_DSNE_PRETRAINED_BACKBONE:-}" ]; then
    DSNE_BACKBONE_ARGS=(--pretrained-backbone "$VDS2RAW_DSNE_PRETRAINED_BACKBONE")
  fi
  for method in "${METHOD_ORDER[@]}"; do
    case "$method" in
      ccsa)
        run_one "ccsa_seed$seed" "$CCSA_MIN_FREE_MIB" \
          "$PYTHON_BIN" "$HERE/vds2raw/train_ccsa.py" \
          --epochs "$CCSA_EPOCHS" --source-epochs "$CCSA_SOURCE_EPOCHS" \
          --batch-size "$CCSA_BATCH" \
          --target-samples-per-class "$CCSA_TARGET_PER_CLASS" \
          "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}" "${PRETRAINED_ARGS[@]}" \
          "${CCSA_CKPT_ARGS[@]}" "${CCSA_BACKBONE_ARGS[@]}"
        ;;
      dsne)
        run_one "dsne_seed$seed" "$DSNE_MIN_FREE_MIB" \
          "$PYTHON_BIN" "$HERE/vds2raw/train_dsne.py" \
          --epochs "$DSNE_EPOCHS" --source-epochs "$DSNE_SOURCE_EPOCHS" \
          --batch-size "$DSNE_BATCH" --nlayers "$DSNE_LAYERS" \
          --target-samples-per-class "$DSNE_TARGET_PER_CLASS" --fn \
          "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}" "${PRETRAINED_ARGS[@]}" \
          "${DSNE_CKPT_ARGS[@]}" "${DSNE_BACKBONE_ARGS[@]}"
        ;;
      mcft3)
        run_one "mcft3_seed$seed" "$MCFT_MIN_FREE_MIB" \
          "$PYTHON_BIN" "$HERE/vds2raw/train_mcft.py" \
          --epochs "$MCFT_EPOCHS" --batch-size "$MCFT_BATCH" \
          --eval-frequency "$MCFT_EVAL" \
          "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}"
        ;;
    esac
  done
done

BENCHMARK_ARGS=()
if [ "$MODE" = smoke ]; then
  BENCHMARK_ARGS=(--smoke)
fi
if [ "${VDS2RAW_RUN_COMPLEXITY_BENCHMARK:-1}" = "1" ]; then
  run_one "complexity" "$MCFT_MIN_FREE_MIB" \
    "$PYTHON_BIN" "$HERE/tools/benchmark_efficiency.py" \
    --methods vds_ccsa,vds_dsne,vds_mcft3 \
    --num-classes 3 --protocol benchmark \
    --benchmark_train_batch_size 1 \
    --efficiency_output_dir "$RUN_DIR/efficiency" \
    "${BENCHMARK_ARGS[@]}"
fi

"$PYTHON_BIN" "$HERE/vds2raw/summarize.py" \
  --results-dir "$ARTIFACT_DIR" --output-dir "$SUMMARY_DIR"
echo "All VDS2Raw methods completed. Summary: $SUMMARY_DIR"

