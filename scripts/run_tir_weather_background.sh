#!/usr/bin/env bash
# One-command background launcher for TIR on every weather target.
# Formal mode runs four targets x five independent seeds (42..46).
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_worker() {
  local mode="$1"
  case "$mode" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;; esac

  local python_bin="${PYTHON:-python}"
  local data_root="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
  local list_dir="${TIR_WEATHER_LIST_DIR:-$HERE/data_lists}"
  local class_names="${TIR_WEATHER_CLASS_NAMES_FILE:-$HERE/class.txt}"
  local clip_checkpoint="${TIR_WEATHER_CLIP_CHECKPOINT:-$HERE/pretrained_weights/clip/ViT-B-16.pt}"
  local output_root="${TIR_WEATHER_OUTPUT_ROOT:-$HERE/runs/tir_weather}"
  local epochs="${TIR_WEATHER_EPOCHS:-100}"
  local batch_size="${TIR_WEATHER_BATCH_SIZE:-32}"
  local num_workers="${TIR_WEATHER_NUM_WORKERS:-2}"
  local min_free_mib="${TIR_WEATHER_MIN_FREE_MIB:-14336}"
  local max_util="${TIR_WEATHER_GPU_MAX_UTIL_PERCENT:-10}"
  local stable_checks="${TIR_WEATHER_GPU_STABLE_CHECKS:-3}"
  local poll_seconds="${TIR_WEATHER_GPU_POLL_SECONDS:-30}"
  local gpu_candidates="${TIR_WEATHER_GPU_CANDIDATES:-}"
  local modality_warmup="${TIR_WEATHER_MODALITY_WARMUP_EPOCHS:-10}"
  local text_unfreeze="${TIR_WEATHER_TEXT_UNFREEZE_EPOCH:-31}"
  local lr_warmup="${TIR_WEATHER_LR_WARMUP_STEPS:-5}"
  local max_batch_args=()
  local targets_text="${TIR_WEATHER_TARGETS:-backlight rainy foggy dark}"
  local seeds_text="${TIR_WEATHER_SEEDS:-42 43 44 45 46}"
  local targets seeds
  read -r -a targets <<< "$targets_text"
  read -r -a seeds <<< "$seeds_text"
  if [ "$mode" = smoke ]; then
    targets=(backlight)
    seeds=(42)
    epochs=1
    batch_size=2
    modality_warmup=0
    text_unfreeze=1
    lr_warmup=0
    max_batch_args=(--max-train-batches 1)
  fi

  export JMDA_DATA_ROOT="$data_root"
  export JMDA_DETERMINISTIC="${JMDA_DETERMINISTIC:-1}"
  if [ ! -f "$clip_checkpoint" ]; then
    echo "Missing TIR CLIP checkpoint: $clip_checkpoint" >&2
    exit 1
  fi
  if [ ! -f "$class_names" ]; then
    echo "Missing class-name mapping: $class_names" >&2
    exit 1
  fi
  "$python_bin" "$HERE/tools/class_names.py" "$class_names" --expected-count 14
  for target in "${targets[@]}"; do
    case "$target" in backlight|rainy|foggy|dark) ;; *) echo "Unknown target: $target" >&2; exit 1 ;; esac
    for split in train val; do
      if [ ! -f "$list_dir/${target}_${split}.txt" ]; then
        echo "Missing weather list: $list_dir/${target}_${split}.txt" >&2
        exit 1
      fi
    done
  done
  if [ ! -f "$list_dir/sunny_train.txt" ]; then
    echo "Missing weather list: $list_dir/sunny_train.txt" >&2
    exit 1
  fi

  local run_id="tir_weather_${mode}_$(date +%Y%m%d_%H%M%S)_$$"
  local run_dir="$output_root/$run_id"
  local artifact_dir="$run_dir/artifacts"
  local log_dir="$run_dir/logs"
  local summary_dir="$run_dir/summary"
  mkdir -p "$artifact_dir" "$log_dir" "$summary_dir"
  echo "Run directory: $run_dir"
  echo "Protocol: sunny source + fully labelled target train; paired VIS+IR; target val selection"
  echo "Targets: ${targets[*]}"
  echo "Independent seeds: ${seeds[*]}"
  echo "Epochs per independent run: $epochs"

  local target seed tag log
  for target in "${targets[@]}"; do
    for seed in "${seeds[@]}"; do
      tag="tir_weather_${target}_seed${seed}"
      log="$log_dir/${tag}.log"
      local gpu_args=(
        --min-free-mib "$min_free_mib"
        --max-util-percent "$max_util"
        --stable-checks "$stable_checks"
        --poll-seconds "$poll_seconds"
        --tag "$tag"
      )
      if [ -n "$gpu_candidates" ]; then gpu_args+=(--gpus "$gpu_candidates"); fi
      echo ">>> $tag (fresh CLIP initialization; log=$log)"
      if "$python_bin" "$HERE/tools/run_on_available_gpu.py" \
          "${gpu_args[@]}" -- \
          "$python_bin" "$HERE/weather/train_tir.py" \
          --target "$target" --seed "$seed" \
          --data-dir "$list_dir" --save-dir "$artifact_dir" \
          --class-names-file "$class_names" \
          --clip-checkpoint "$clip_checkpoint" \
          --epochs "$epochs" --batch-size "$batch_size" \
          --eval-frequency 1 --num-workers "$num_workers" \
          --lora-lr "${TIR_WEATHER_LORA_LR:-0.0002}" \
          --text-lora-lr-scale "${TIR_WEATHER_TEXT_LORA_LR_SCALE:-0.25}" \
          --prompt-lr "${TIR_WEATHER_PROMPT_LR:-0.0005}" \
          --adapter-lr "${TIR_WEATHER_ADAPTER_LR:-0.001}" \
          --gate-lr "${TIR_WEATHER_GATE_LR:-0.002}" \
          --lora-weight-decay "${TIR_WEATHER_LORA_WEIGHT_DECAY:-0.01}" \
          --weight-decay "${TIR_WEATHER_WEIGHT_DECAY:-0.0005}" \
          --lora-rank "${TIR_WEATHER_LORA_RANK:-16}" \
          --lora-alpha "${TIR_WEATHER_LORA_ALPHA:-8}" \
          --lora-dropout "${TIR_WEATHER_LORA_DROPOUT:-0.25}" \
          --lora-encoder both --lora-params q k v \
          --tir-min-weight "${TIR_WEATHER_MIN_TOKEN_WEIGHT:-0.5}" \
          --tir-max-weight "${TIR_WEATHER_MAX_TOKEN_WEIGHT:-1.5}" \
          --ir-initial-scale "${TIR_WEATHER_IR_INITIAL_SCALE:-0.1}" \
          --ir-dropout "${TIR_WEATHER_IR_DROPOUT:-0.1}" \
          --ir-aux-loss-weight "${TIR_WEATHER_IR_AUX_WEIGHT:-0.2}" \
          --class-weight-power "${TIR_WEATHER_CLASS_WEIGHT_POWER:-0.5}" \
          --source-loss-weight "${TIR_WEATHER_SOURCE_LOSS_WEIGHT:-0.5}" \
          --target-loss-weight "${TIR_WEATHER_TARGET_LOSS_WEIGHT:-0.5}" \
          --modality-warmup-epochs "$modality_warmup" \
          --text-unfreeze-epoch "$text_unfreeze" \
          --lr-warmup-steps "$lr_warmup" \
          --min-lr-ratio "${TIR_WEATHER_MIN_LR_RATIO:-0.1}" \
          --grad-clip "${TIR_WEATHER_GRAD_CLIP:-1.0}" \
          --early-stopping-patience "${TIR_WEATHER_EARLY_STOPPING_PATIENCE:-0}" \
          "${max_batch_args[@]}" > "$log" 2>&1; then
        echo "    completed: $tag"
      else
        tail -n 60 "$log" || true
        echo "FAILED: $tag; log=$log" >&2
        exit 1
      fi
    done
  done
  "$python_bin" "$HERE/tools/summarize_tir_weather.py" \
    --results-dir "$artifact_dir" --output-dir "$summary_dir"
  echo "All TIR weather runs completed. Summary: $summary_dir"
}

if [ "${1:-}" = "__worker" ]; then
  run_worker "${2:-formal}"
  exit 0
fi

MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;; esac
LOG_ROOT="$HERE/server_logs/tir_weather_launcher"
mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_ROOT/${MODE}_${STAMP}.log"
PID_FILE="$LOG_ROOT/${MODE}_${STAMP}.pid"
SELF="$HERE/scripts/run_tir_weather_background.sh"
nohup bash "$SELF" __worker "$MODE" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "PID=$PID"
echo "log=$LOG"
echo "pid_file=$PID_FILE"
echo "tail -f '$LOG'"
