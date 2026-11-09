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
read -r -a METHOD_ORDER <<< "${VDS2RAW_METHOD_ORDER:-tir}"

MCFT_EPOCHS="${VDS2RAW_MCFT_EPOCHS:-200}"
MCFT_BATCH="${VDS2RAW_MCFT_BATCH_SIZE:-32}"
MCFT_LR="${VDS2RAW_MCFT_LR:-0.0005}"
MCFT_WEIGHT_DECAY="${VDS2RAW_MCFT_WEIGHT_DECAY:-0.0001}"
MCFT_STEP_SIZE="${VDS2RAW_MCFT_STEP_SIZE:-20}"
MCFT_GAMMA="${VDS2RAW_MCFT_GAMMA:-0.9}"
MCFT_LABEL_SMOOTHING="${VDS2RAW_MCFT_LABEL_SMOOTHING:-0.05}"
MCFT_ITA_WEIGHT="${VDS2RAW_MCFT_ITA_WEIGHT:-0.05}"
MCFT_ITM_WEIGHT="${VDS2RAW_MCFT_ITM_WEIGHT:-1.0}"
MCFT_AIS_DROPOUT="${VDS2RAW_MCFT_AIS_DROPOUT:-0.3}"
MCFT_EARLY_PATIENCE="${VDS2RAW_MCFT_EARLY_STOPPING_PATIENCE:-25}"
MCFT_EARLY_MIN_EPOCHS="${VDS2RAW_MCFT_EARLY_STOPPING_MIN_EPOCHS:-60}"

TIR_EPOCHS="${VDS2RAW_TIR_EPOCHS:-39}"
TIR_BATCH="${VDS2RAW_TIR_BATCH_SIZE:-32}"
TIR_LORA_LR="${VDS2RAW_TIR_LORA_LR:-0.0002}"
TIR_TEXT_LORA_LR_SCALE="${VDS2RAW_TIR_TEXT_LORA_LR_SCALE:-0.25}"
TIR_PROMPT_LR="${VDS2RAW_TIR_PROMPT_LR:-0.0005}"
TIR_ADAPTER_LR="${VDS2RAW_TIR_ADAPTER_LR:-0.001}"
TIR_GATE_LR="${VDS2RAW_TIR_GATE_LR:-0.002}"
TIR_LORA_WEIGHT_DECAY="${VDS2RAW_TIR_LORA_WEIGHT_DECAY:-0.01}"
TIR_TASK_WEIGHT_DECAY="${VDS2RAW_TIR_TASK_WEIGHT_DECAY:-0.0005}"
TIR_LORA_RANK="${VDS2RAW_TIR_LORA_RANK:-16}"
TIR_LORA_ALPHA="${VDS2RAW_TIR_LORA_ALPHA:-8}"
TIR_LORA_DROPOUT="${VDS2RAW_TIR_LORA_DROPOUT:-0.25}"
TIR_LABEL_SMOOTHING="${VDS2RAW_TIR_LABEL_SMOOTHING:-0.0}"
TIR_AIS_DROPOUT="${VDS2RAW_TIR_AIS_DROPOUT:-0.1}"
TIR_NIR_INITIAL_SCALE="${VDS2RAW_TIR_NIR_INITIAL_SCALE:-0.05}"
TIR_AIS_INITIAL_SCALE="${VDS2RAW_TIR_AIS_INITIAL_SCALE:-0.75}"
TIR_MODALITY_WARMUP_EPOCHS="${VDS2RAW_TIR_MODALITY_WARMUP_EPOCHS:-25}"
TIR_TEXT_UNFREEZE_EPOCH="${VDS2RAW_TIR_TEXT_UNFREEZE_EPOCH:-40}"
TIR_NIR_AUX_WEIGHT="${VDS2RAW_TIR_NIR_AUX_LOSS_WEIGHT:-0.2}"
TIR_AIS_AUX_WEIGHT="${VDS2RAW_TIR_AIS_AUX_LOSS_WEIGHT:-0.2}"
TIR_MIN_TOKEN_WEIGHT="${VDS2RAW_TIR_MIN_TOKEN_WEIGHT:-0.5}"
TIR_MAX_TOKEN_WEIGHT="${VDS2RAW_TIR_MAX_TOKEN_WEIGHT:-1.5}"
TIR_CLASS_WEIGHT_POWER="${VDS2RAW_TIR_CLASS_WEIGHT_POWER:-0.5}"
TIR_SOURCE_LOSS_WEIGHT="${VDS2RAW_TIR_SOURCE_LOSS_WEIGHT:-0.4}"
TIR_TARGET_LOSS_WEIGHT="${VDS2RAW_TIR_TARGET_LOSS_WEIGHT:-0.6}"
TIR_MIN_LR_RATIO="${VDS2RAW_TIR_MIN_LR_RATIO:-0.1}"
TIR_SCHEDULER="${VDS2RAW_TIR_SCHEDULER:-cosine}"
TIR_WARMUP_EPOCHS="${VDS2RAW_TIR_WARMUP_EPOCHS:-5}"
TIR_SCHEDULER_HORIZON="${VDS2RAW_TIR_SCHEDULER_HORIZON_EPOCHS:-100}"
TIR_EARLY_PATIENCE="${VDS2RAW_TIR_EARLY_STOPPING_PATIENCE:-0}"
TIR_EARLY_MIN_EPOCHS="${VDS2RAW_TIR_EARLY_STOPPING_MIN_EPOCHS:-60}"
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
# Reject invalid AIS gates before auditing data or waiting for a GPU.  The
# model reparameterises this positive initial value with softplus, so the
# effective gate remains strictly positive while it is trained.
if [ "$seen_tir" -gt 0 ] && ! "$PYTHON_BIN" -c \
    'import math, sys; value = float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0.0 else 1)' \
    "$TIR_AIS_INITIAL_SCALE"; then
  echo "VDS2RAW_TIR_AIS_INITIAL_SCALE must be finite and strictly positive: $TIR_AIS_INITIAL_SCALE" >&2
  exit 1
fi
if [ "$seen_tir" -gt 0 ] && [ ! -f "$TIR_CLIP_CHECKPOINT" ]; then
  echo "Missing TIR CLIP checkpoint: $TIR_CLIP_CHECKPOINT" >&2
  exit 1
fi

MCFT_MAX_BATCH_ARGS=()
TIR_MAX_BATCH_ARGS=()
if [ "$MODE" = smoke ]; then
  MCFT_EPOCHS=1; TIR_EPOCHS=1
  MCFT_BATCH=2; TIR_BATCH=32
  MCFT_EARLY_PATIENCE=0; TIR_EARLY_PATIENCE=0
  TIR_WARMUP_EPOCHS=0
  TIR_MODALITY_WARMUP_EPOCHS=0
  TIR_TEXT_UNFREEZE_EPOCH=1
  MCFT_MAX_BATCH_ARGS=(--max-train-batches 1)
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
if [ "$seen_tir" -gt 0 ]; then
  echo "TIR AIS initial weight: $TIR_AIS_INITIAL_SCALE (positive confidence fusion)"
fi
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
          --loss-ita-weight "$MCFT_ITA_WEIGHT" \
          --loss-itm-weight "$MCFT_ITM_WEIGHT" \
          --ais-dropout-probability "$MCFT_AIS_DROPOUT" \
          --early-stopping-patience "$MCFT_EARLY_PATIENCE" \
          --early-stopping-min-epochs "$MCFT_EARLY_MIN_EPOCHS" \
          --stats-file "$STATS_FILE" \
          "${COMMON[@]}" "${MCFT_MAX_BATCH_ARGS[@]}"
        ;;
      tir)
        run_one "tir_seed$seed" "$TIR_MIN_FREE_MIB" \
          "$PYTHON_BIN" "$HERE/vds2raw/train_tir.py" \
          --epochs "$TIR_EPOCHS" --batch-size "$TIR_BATCH" \
          --eval-frequency 1 \
          --clip-checkpoint "$TIR_CLIP_CHECKPOINT" \
          --lora-lr "$TIR_LORA_LR" \
          --text-lora-lr-scale "$TIR_TEXT_LORA_LR_SCALE" \
          --prompt-lr "$TIR_PROMPT_LR" \
          --adapter-lr "$TIR_ADAPTER_LR" --gate-lr "$TIR_GATE_LR" \
          --lora-weight-decay "$TIR_LORA_WEIGHT_DECAY" \
          --weight-decay "$TIR_TASK_WEIGHT_DECAY" \
          --lora-rank "$TIR_LORA_RANK" --lora-alpha "$TIR_LORA_ALPHA" \
          --lora-dropout "$TIR_LORA_DROPOUT" \
          --lora-encoder both --lora-params q k v \
          --ais-dropout-probability "$TIR_AIS_DROPOUT" \
          --nir-initial-scale "$TIR_NIR_INITIAL_SCALE" \
          --ais-initial-scale "$TIR_AIS_INITIAL_SCALE" \
          --modality-warmup-epochs "$TIR_MODALITY_WARMUP_EPOCHS" \
          --text-unfreeze-epoch "$TIR_TEXT_UNFREEZE_EPOCH" \
          --nir-aux-loss-weight "$TIR_NIR_AUX_WEIGHT" \
          --ais-aux-loss-weight "$TIR_AIS_AUX_WEIGHT" \
          --tir-min-weight "$TIR_MIN_TOKEN_WEIGHT" \
          --tir-max-weight "$TIR_MAX_TOKEN_WEIGHT" \
          --class-weight-power "$TIR_CLASS_WEIGHT_POWER" \
          --source-loss-weight "$TIR_SOURCE_LOSS_WEIGHT" \
          --target-loss-weight "$TIR_TARGET_LOSS_WEIGHT" \
          --label-smoothing "$TIR_LABEL_SMOOTHING" \
          --scheduler "$TIR_SCHEDULER" --warmup-epochs "$TIR_WARMUP_EPOCHS" \
          --scheduler-horizon-epochs "$TIR_SCHEDULER_HORIZON" \
          --min-lr-ratio "$TIR_MIN_LR_RATIO" --augment \
          --natural-train-sampling --minibatch-updates \
          --early-stopping-patience "$TIR_EARLY_PATIENCE" \
          --early-stopping-min-epochs "$TIR_EARLY_MIN_EPOCHS" \
          --stats-file "$STATS_FILE" \
          "${COMMON[@]}" "${TIR_MAX_BATCH_ARGS[@]}"
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
