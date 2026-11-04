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
MCFT_EPOCHS="${VDS2RAW_MCFT_EPOCHS:-200}"
CCSA_SOURCE_EPOCHS="${VDS2RAW_CCSA_SOURCE_EPOCHS:-50}"
DSNE_SOURCE_EPOCHS="${VDS2RAW_DSNE_SOURCE_EPOCHS:-100}"
CCSA_BATCH="${VDS2RAW_CCSA_BATCH_SIZE:-256}"
CCSA_SOURCE_BATCH="${VDS2RAW_CCSA_SOURCE_BATCH_SIZE:-32}"
DSNE_BATCH="${VDS2RAW_DSNE_BATCH_SIZE:-32}"
DSNE_SOURCE_BATCH="${VDS2RAW_DSNE_SOURCE_BATCH_SIZE:-16}"
MCFT_BATCH="${VDS2RAW_MCFT_BATCH_SIZE:-32}"
CCSA_LR="${VDS2RAW_CCSA_LR:-0.005}"
CCSA_BACKBONE_LR_MULTIPLIER="${VDS2RAW_CCSA_BACKBONE_LR_MULTIPLIER:-0.02}"
CCSA_SOURCE_LR="${VDS2RAW_CCSA_SOURCE_LR:-0.5}"
CCSA_LR_MIN_RATIO="${VDS2RAW_CCSA_LR_MIN_RATIO:-0.01}"
CCSA_SOURCE_LR_MIN_RATIO="${VDS2RAW_CCSA_SOURCE_LR_MIN_RATIO:-0.1}"
DSNE_LAYERS="${VDS2RAW_DSNE_LAYERS:-101}"
DSNE_SOURCE_WARMUP="${VDS2RAW_DSNE_SOURCE_HEAD_WARMUP_EPOCHS:-10}"
DSNE_LR="${VDS2RAW_DSNE_LR:-0.000125}"
DSNE_ADAPT_BACKBONE_LR_MULTIPLIER="${VDS2RAW_DSNE_ADAPTATION_BACKBONE_LR_MULTIPLIER:-0.05}"
DSNE_LR_MIN_RATIO="${VDS2RAW_DSNE_LR_MIN_RATIO:-0.1}"
DSNE_SOURCE_HEAD_LR="${VDS2RAW_DSNE_SOURCE_HEAD_LR:-0.005}"
DSNE_SOURCE_BACKBONE_LR="${VDS2RAW_DSNE_SOURCE_BACKBONE_LR:-0.0005}"
DSNE_SOURCE_LR_MIN_RATIO="${VDS2RAW_DSNE_SOURCE_LR_MIN_RATIO:-0.1}"
DSNE_WEIGHT_DECAY="${VDS2RAW_DSNE_WEIGHT_DECAY:-0.0001}"
DSNE_ALPHA="${VDS2RAW_DSNE_ALPHA:-0.2}"
DSNE_LABEL_SMOOTHING="${VDS2RAW_DSNE_LABEL_SMOOTHING:-0.05}"
MCFT_LR="${VDS2RAW_MCFT_LR:-0.0005}"
MCFT_WEIGHT_DECAY="${VDS2RAW_MCFT_WEIGHT_DECAY:-0.0001}"
MCFT_STEP_SIZE="${VDS2RAW_MCFT_STEP_SIZE:-20}"
MCFT_GAMMA="${VDS2RAW_MCFT_GAMMA:-0.9}"
MCFT_LABEL_SMOOTHING="${VDS2RAW_MCFT_LABEL_SMOOTHING:-0.1}"
MCFT_ITM_WEIGHT="${VDS2RAW_MCFT_ITM_WEIGHT:-1.0}"
MCFT_AIS_DROPOUT="${VDS2RAW_MCFT_AIS_DROPOUT_PROBABILITY:-0.5}"
MCFT_CLASS_BALANCED="${VDS2RAW_MCFT_CLASS_BALANCED_TRAIN:-0}"
CCSA_EARLY_PATIENCE="${VDS2RAW_CCSA_EARLY_STOPPING_PATIENCE:-20}"
CCSA_EARLY_MIN_EPOCHS="${VDS2RAW_CCSA_EARLY_STOPPING_MIN_EPOCHS:-30}"
DSNE_EARLY_PATIENCE="${VDS2RAW_DSNE_EARLY_STOPPING_PATIENCE:-10}"
DSNE_EARLY_MIN_EPOCHS="${VDS2RAW_DSNE_EARLY_STOPPING_MIN_EPOCHS:-20}"
MCFT_EARLY_PATIENCE="${VDS2RAW_MCFT_EARLY_STOPPING_PATIENCE:-25}"
MCFT_EARLY_MIN_EPOCHS="${VDS2RAW_MCFT_EARLY_STOPPING_MIN_EPOCHS:-60}"
LEGACY_TARGET_PER_CLASS="${VDS2RAW_TARGET_SAMPLES_PER_CLASS:-}"
CCSA_TARGET_PER_CLASS="${VDS2RAW_CCSA_TARGET_SAMPLES_PER_CLASS:-${LEGACY_TARGET_PER_CLASS:-0}}"
DSNE_TARGET_PER_CLASS="${VDS2RAW_DSNE_TARGET_SAMPLES_PER_CLASS:-${LEGACY_TARGET_PER_CLASS:-0}}"
MCFT_EVAL="${VDS2RAW_MCFT_EVAL_FREQUENCY:-1}"
GPU_MAX_UTIL="${VDS2RAW_GPU_MAX_UTIL_PERCENT:-10}"
GPU_STABLE_CHECKS="${VDS2RAW_GPU_STABLE_CHECKS:-3}"
GPU_POLL_SECONDS="${VDS2RAW_GPU_POLL_SECONDS:-30}"
GPU_CANDIDATES="${VDS2RAW_GPU_CANDIDATES:-}"
CCSA_MIN_FREE_MIB="${VDS2RAW_CCSA_MIN_FREE_MIB:-7168}"
DSNE_MIN_FREE_MIB="${VDS2RAW_DSNE_MIN_FREE_MIB:-12288}"
MCFT_MIN_FREE_MIB="${VDS2RAW_MCFT_MIN_FREE_MIB:-4608}"
case "$MCFT_CLASS_BALANCED" in
  0) MCFT_BALANCE_ARGS=(--no-class-balanced-train) ;;
  1) MCFT_BALANCE_ARGS=(--class-balanced-train) ;;
  *) echo "VDS2RAW_MCFT_CLASS_BALANCED_TRAIN must be 0 or 1." >&2; exit 1 ;;
esac
if [ "${#METHOD_ORDER[@]}" -lt 1 ] || [ "${#METHOD_ORDER[@]}" -gt 3 ]; then
  echo "VDS2RAW_METHOD_ORDER must contain one to three methods from: ccsa dsne mcft3." >&2
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
if [ "$CCSA_ORDER_COUNT" -gt 1 ] || [ "$DSNE_ORDER_COUNT" -gt 1 ] \
  || [ "$MCFT_ORDER_COUNT" -gt 1 ]; then
  echo "VDS2RAW_METHOD_ORDER cannot contain duplicate methods." >&2
  exit 1
fi
if [ "$MODE" = formal ] && \
   { { [ "$CCSA_ORDER_COUNT" -gt 0 ] && [ "$CCSA_TARGET_PER_CLASS" -ne 0 ]; } \
     || { [ "$DSNE_ORDER_COUNT" -gt 0 ] && [ "$DSNE_TARGET_PER_CLASS" -ne 0 ]; }; }; then
  echo "Formal VDS2Raw runs require all target_train labels: set both target-samples-per-class values to 0." >&2
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
  CCSA_BATCH=8; CCSA_SOURCE_BATCH=4; DSNE_BATCH=4; DSNE_SOURCE_BATCH=4; MCFT_BATCH=4
  DSNE_LAYERS=101; DSNE_SOURCE_WARMUP=0; MCFT_EVAL=1
  CCSA_EARLY_PATIENCE=0; DSNE_EARLY_PATIENCE=0; MCFT_EARLY_PATIENCE=0
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
          --batch-size "$CCSA_BATCH" --source-batch-size "$CCSA_SOURCE_BATCH" \
          --lr "$CCSA_LR" --source-lr "$CCSA_SOURCE_LR" \
          --backbone-lr-multiplier "$CCSA_BACKBONE_LR_MULTIPLIER" \
          --lr-min-ratio "$CCSA_LR_MIN_RATIO" \
          --source-lr-min-ratio "$CCSA_SOURCE_LR_MIN_RATIO" \
          --class-balanced-ce \
          --early-stopping-patience "$CCSA_EARLY_PATIENCE" \
          --early-stopping-min-epochs "$CCSA_EARLY_MIN_EPOCHS" \
          --target-samples-per-class "$CCSA_TARGET_PER_CLASS" \
          "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}" "${PRETRAINED_ARGS[@]}" \
          "${CCSA_CKPT_ARGS[@]}" "${CCSA_BACKBONE_ARGS[@]}"
        ;;
      dsne)
        run_one "dsne_seed$seed" "$DSNE_MIN_FREE_MIB" \
          "$PYTHON_BIN" "$HERE/vds2raw/train_dsne.py" \
          --epochs "$DSNE_EPOCHS" --source-epochs "$DSNE_SOURCE_EPOCHS" \
          --batch-size "$DSNE_BATCH" --source-batch-size "$DSNE_SOURCE_BATCH" \
          --lr "$DSNE_LR" --lr-min-ratio "$DSNE_LR_MIN_RATIO" \
          --adaptation-backbone-lr-multiplier \
          "$DSNE_ADAPT_BACKBONE_LR_MULTIPLIER" \
          --source-head-warmup-epochs "$DSNE_SOURCE_WARMUP" \
          --source-head-lr "$DSNE_SOURCE_HEAD_LR" \
          --source-backbone-lr "$DSNE_SOURCE_BACKBONE_LR" \
          --source-lr-min-ratio "$DSNE_SOURCE_LR_MIN_RATIO" \
          --weight-decay "$DSNE_WEIGHT_DECAY" --alpha "$DSNE_ALPHA" \
          --label-smoothing "$DSNE_LABEL_SMOOTHING" --dropout \
          --early-stopping-patience "$DSNE_EARLY_PATIENCE" \
          --early-stopping-min-epochs "$DSNE_EARLY_MIN_EPOCHS" \
          --augmentation-strength light --class-balanced-source \
          --class-balanced-batches --nlayers "$DSNE_LAYERS" \
          --target-samples-per-class "$DSNE_TARGET_PER_CLASS" --fn \
          "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}" "${PRETRAINED_ARGS[@]}" \
          "${DSNE_CKPT_ARGS[@]}" "${DSNE_BACKBONE_ARGS[@]}"
        ;;
      mcft3)
        run_one "mcft3_seed$seed" "$MCFT_MIN_FREE_MIB" \
          "$PYTHON_BIN" "$HERE/vds2raw/train_mcft.py" \
          --epochs "$MCFT_EPOCHS" --batch-size "$MCFT_BATCH" \
          --lr "$MCFT_LR" --weight-decay "$MCFT_WEIGHT_DECAY" \
          --step-size "$MCFT_STEP_SIZE" --gamma "$MCFT_GAMMA" \
          --label-smoothing "$MCFT_LABEL_SMOOTHING" \
          --loss-itm-weight "$MCFT_ITM_WEIGHT" \
          --ais-dropout-probability "$MCFT_AIS_DROPOUT" \
          "${MCFT_BALANCE_ARGS[@]}" \
          --eval-frequency "$MCFT_EVAL" \
          --early-stopping-patience "$MCFT_EARLY_PATIENCE" \
          --early-stopping-min-epochs "$MCFT_EARLY_MIN_EPOCHS" \
          "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}"
        ;;
    esac
  done
done

BENCHMARK_ARGS=()
BENCHMARK_METHODS=()
for method in "${METHOD_ORDER[@]}"; do
  case "$method" in
    ccsa) BENCHMARK_METHODS+=(vds_ccsa) ;;
    dsne) BENCHMARK_METHODS+=(vds_dsne) ;;
    mcft3) BENCHMARK_METHODS+=(vds_mcft3) ;;
  esac
done
BENCHMARK_METHODS_CSV="$(IFS=,; echo "${BENCHMARK_METHODS[*]}")"
if [ "$MODE" = smoke ]; then
  BENCHMARK_ARGS=(--smoke)
fi
if [ "${VDS2RAW_RUN_COMPLEXITY_BENCHMARK:-1}" = "1" ]; then
  run_one "complexity" "$MCFT_MIN_FREE_MIB" \
    "$PYTHON_BIN" "$HERE/tools/benchmark_efficiency.py" \
    --methods "$BENCHMARK_METHODS_CSV" \
    --num-classes 3 --protocol benchmark \
    --benchmark_train_batch_size 1 \
    --efficiency_output_dir "$RUN_DIR/efficiency" \
    "${BENCHMARK_ARGS[@]}"
fi

"$PYTHON_BIN" "$HERE/vds2raw/summarize.py" \
  --results-dir "$ARTIFACT_DIR" --output-dir "$SUMMARY_DIR"
echo "All VDS2Raw methods completed. Summary: $SUMMARY_DIR"

