#!/usr/bin/env bash
# Sequential VDS2Raw runner for the retained MCFT and new TIR experiments.
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;; esac

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${VDS2RAW_DATA_ROOT:-/home/lixiang/lx/VDS2Raw/threeclass_ready}"
OUTPUT_ROOT="${VDS2RAW_OUTPUT_ROOT:-$HERE/runs/vds2raw}"
read -r -a SEEDS <<< "${VDS2RAW_SEEDS:-42}"
read -r -a METHOD_ORDER <<< "${VDS2RAW_METHOD_ORDER:-mcft3 tir}"

MCFT_EPOCHS="${VDS2RAW_MCFT_EPOCHS:-200}"
MCFT_BATCH="${VDS2RAW_MCFT_BATCH_SIZE:-32}"
MCFT_LR="${VDS2RAW_MCFT_LR:-0.0005}"
MCFT_WEIGHT_DECAY="${VDS2RAW_MCFT_WEIGHT_DECAY:-0.0001}"
MCFT_STEP_SIZE="${VDS2RAW_MCFT_STEP_SIZE:-20}"
MCFT_GAMMA="${VDS2RAW_MCFT_GAMMA:-0.9}"
MCFT_LABEL_SMOOTHING="${VDS2RAW_MCFT_LABEL_SMOOTHING:-0.1}"
MCFT_ITM_WEIGHT="${VDS2RAW_MCFT_ITM_WEIGHT:-1.0}"
MCFT_AIS_DROPOUT="${VDS2RAW_MCFT_AIS_DROPOUT:-0.5}"
MCFT_EARLY_PATIENCE="${VDS2RAW_MCFT_EARLY_STOPPING_PATIENCE:-25}"
MCFT_EARLY_MIN_EPOCHS="${VDS2RAW_MCFT_EARLY_STOPPING_MIN_EPOCHS:-60}"

TIR_EPOCHS="${VDS2RAW_TIR_EPOCHS:-100}"
TIR_BATCH="${VDS2RAW_TIR_BATCH_SIZE:-9999}"
TIR_SHOTS="${VDS2RAW_TIR_SHOTS_PER_CLASS:-5}"
TIR_LORA_LR="${VDS2RAW_TIR_LORA_LR:-0.0002}"
TIR_PROMPT_LR="${VDS2RAW_TIR_PROMPT_LR:-0.002}"
TIR_ADAPTER_LR="${VDS2RAW_TIR_ADAPTER_LR:-0.001}"
TIR_LORA_WEIGHT_DECAY="${VDS2RAW_TIR_LORA_WEIGHT_DECAY:-0.01}"
TIR_TASK_WEIGHT_DECAY="${VDS2RAW_TIR_TASK_WEIGHT_DECAY:-0.0005}"
TIR_EARLY_PATIENCE="${VDS2RAW_TIR_EARLY_STOPPING_PATIENCE:-0}"
TIR_EARLY_MIN_EPOCHS="${VDS2RAW_TIR_EARLY_STOPPING_MIN_EPOCHS:-0}"
TIR_CLIP_CHECKPOINT="${VDS2RAW_TIR_CLIP_CHECKPOINT:-$HERE/pretrained_weights/clip/ViT-B-16.pt}"

GPU_CANDIDATES="${VDS2RAW_GPU_CANDIDATES:-}"
GPU_MAX_UTIL="${VDS2RAW_GPU_MAX_UTIL_PERCENT:-10}"
GPU_STABLE_CHECKS="${VDS2RAW_GPU_STABLE_CHECKS:-3}"
GPU_POLL_SECONDS="${VDS2RAW_GPU_POLL_SECONDS:-30}"
MCFT_MIN_FREE_MIB="${VDS2RAW_MCFT_MIN_FREE_MIB:-6144}"
TIR_MIN_FREE_MIB="${VDS2RAW_TIR_MIN_FREE_MIB:-14336}"

if [ "${#METHOD_ORDER[@]}" -lt 1 ] || [ "${#METHOD_ORDER[@]}" -gt 2 ]; then
  echo "VDS2RAW_METHOD_ORDER must contain one or two methods from: mcft3 tir." >&2
  exit 1
fi
seen_mcft=0; seen_tir=0
for method in "${METHOD_ORDER[@]}"; do
  case "$method" in
    mcft3) seen_mcft=$((seen_mcft + 1)) ;;
    tir) seen_tir=$((seen_tir + 1)) ;;
    *) echo "Unknown VDS2Raw method: $method" >&2; exit 1 ;;
  esac
done
if [ "$seen_mcft" -gt 1 ] || [ "$seen_tir" -gt 1 ]; then
  echo "VDS2RAW_METHOD_ORDER cannot contain duplicates." >&2
  exit 1
fi
if [ "$seen_tir" -gt 0 ] && [ ! -f "$TIR_CLIP_CHECKPOINT" ]; then
  echo "Missing TIR CLIP checkpoint: $TIR_CLIP_CHECKPOINT" >&2
  exit 1
fi

MAX_BATCH_ARGS=()
if [ "$MODE" = smoke ]; then
  MCFT_EPOCHS=1; TIR_EPOCHS=1
  MCFT_BATCH=2; TIR_BATCH=2
  MCFT_EARLY_PATIENCE=0; TIR_EARLY_PATIENCE=0
  MAX_BATCH_ARGS=(--max-train-batches 1)
fi

RUN_ID="vds2raw_${MODE}_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
ARTIFACT_DIR="$RUN_DIR/artifacts"
LOG_DIR="$RUN_DIR/logs"
SUMMARY_DIR="$RUN_DIR/summary"
STATS_FILE="$RUN_DIR/normalization_stats.json"
TIR_STATS_FILE="$RUN_DIR/tir_target_normalization_stats.json"
mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$SUMMARY_DIR"

echo "Run directory: $RUN_DIR"
echo "Method order: ${METHOD_ORDER[*]}"
"$PYTHON_BIN" "$HERE/vds2raw/audit.py" --data-root "$DATA_ROOT" \
  --stats-file "$STATS_FILE" --report-file "$RUN_DIR/dataset_audit.json"

run_one() {
  local tag="$1" min_free_mib="$2"; shift 2
  local log="$LOG_DIR/$tag.log"
  local gpu_args=(--min-free-mib "$min_free_mib" \
    --max-util-percent "$GPU_MAX_UTIL" --stable-checks "$GPU_STABLE_CHECKS" \
    --poll-seconds "$GPU_POLL_SECONDS" --tag "$tag")
  if [ -n "$GPU_CANDIDATES" ]; then gpu_args+=(--gpus "$GPU_CANDIDATES"); fi
  echo ">>> $tag (GPU gate: free>=${min_free_mib} MiB, util<=${GPU_MAX_UTIL}%; log=$log)"
  if "$PYTHON_BIN" "$HERE/tools/run_on_available_gpu.py" \
      "${gpu_args[@]}" -- "$@" > "$log" 2>&1; then
    echo "    completed; log: $log"
  else
    tail -n 40 "$log" || true
    echo "FAILED: $tag; log: $log" >&2
    exit 1
  fi
}

for seed in "${SEEDS[@]}"; do
  COMMON=(--data-root "$DATA_ROOT" --save-dir "$ARTIFACT_DIR" \
    --seed "$seed" --num-workers "${VDS2RAW_NUM_WORKERS:-2}" --skip-audit)
  for method in "${METHOD_ORDER[@]}"; do
    case "$method" in
      mcft3)
        run_one "mcft3_seed$seed" "$MCFT_MIN_FREE_MIB" \
          "$PYTHON_BIN" "$HERE/vds2raw/train_mcft.py" \
          --epochs "$MCFT_EPOCHS" --batch-size "$MCFT_BATCH" \
          --lr "$MCFT_LR" --weight-decay "$MCFT_WEIGHT_DECAY" \
          --step-size "$MCFT_STEP_SIZE" --gamma "$MCFT_GAMMA" \
          --label-smoothing "$MCFT_LABEL_SMOOTHING" \
          --loss-itm-weight "$MCFT_ITM_WEIGHT" \
          --ais-dropout-probability "$MCFT_AIS_DROPOUT" \
          --early-stopping-patience "$MCFT_EARLY_PATIENCE" \
          --early-stopping-min-epochs "$MCFT_EARLY_MIN_EPOCHS" \
          --stats-file "$STATS_FILE" \
          "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}"
        ;;
      tir)
        run_one "tir_seed$seed" "$TIR_MIN_FREE_MIB" \
          "$PYTHON_BIN" "$HERE/vds2raw/train_tir.py" \
          --epochs "$TIR_EPOCHS" --batch-size "$TIR_BATCH" \
          --shots-per-class "$TIR_SHOTS" --eval-frequency "$TIR_EPOCHS" \
          --clip-checkpoint "$TIR_CLIP_CHECKPOINT" \
          --lora-lr "$TIR_LORA_LR" --prompt-lr "$TIR_PROMPT_LR" \
          --adapter-lr "$TIR_ADAPTER_LR" \
          --lora-weight-decay "$TIR_LORA_WEIGHT_DECAY" \
          --weight-decay "$TIR_TASK_WEIGHT_DECAY" \
          --early-stopping-patience "$TIR_EARLY_PATIENCE" \
          --early-stopping-min-epochs "$TIR_EARLY_MIN_EPOCHS" \
          --stats-file "$TIR_STATS_FILE" \
          "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}"
        ;;
    esac
  done
done

if [ "${VDS2RAW_RUN_COMPLEXITY_BENCHMARK:-1}" = "1" ]; then
  BENCHMARK_METHODS=()
  for method in "${METHOD_ORDER[@]}"; do
    case "$method" in mcft3) BENCHMARK_METHODS+=(vds_mcft3) ;; tir) BENCHMARK_METHODS+=(vds_tir) ;; esac
  done
  BENCHMARK_CSV="$(IFS=,; echo "${BENCHMARK_METHODS[*]}")"
  BENCHMARK_ARGS=(); if [ "$MODE" = smoke ]; then BENCHMARK_ARGS=(--smoke); fi
  run_one "complexity" "$TIR_MIN_FREE_MIB" \
    "$PYTHON_BIN" "$HERE/tools/benchmark_efficiency.py" \
    --methods "$BENCHMARK_CSV" --num-classes 3 --protocol benchmark \
    --benchmark_train_batch_size 1 --efficiency_output_dir "$RUN_DIR/efficiency" \
    "${BENCHMARK_ARGS[@]}"
fi

"$PYTHON_BIN" "$HERE/vds2raw/summarize.py" \
  --results-dir "$ARTIFACT_DIR" --output-dir "$SUMMARY_DIR"
echo "All VDS2Raw methods completed. Summary: $SUMMARY_DIR"
