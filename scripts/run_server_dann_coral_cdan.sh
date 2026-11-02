#!/usr/bin/env bash
# Run the multimodal DANN, DeepCORAL and CDAN server suite on physical GPU 0/1.
# A job is launched only after one candidate GPU has enough free memory and a
# sufficiently low GPU-util reading for several consecutive nvidia-smi polls.

set -uo pipefail
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
TARGET_TEXT="${JMDA_TARGETS:-backlight rainy foggy dark}"
SEED="${JMDA_SEED:-42}"
DANN_EPOCHS="${DANN_EPOCHS:-100}"
DEEPCORAL_EPOCHS="${DEEPCORAL_EPOCHS:-20}"
DANN_ITERATIONS="${DANN_ITERATIONS:-0}"
DEEPCORAL_ITERATIONS="${DEEPCORAL_ITERATIONS:-0}"
CDAN_ITERATIONS="${CDAN_ITERATIONS:-2000}"
BATCH_SIZE="${UDA_IMAGE_BATCH_SIZE:-16}"
NUM_WORKERS="${UDA_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${UDA_PREFETCH_FACTOR:-4}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0,1}"
MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-18000}"
DANN_MIN_FREE_MIB="${DANN_MIN_FREE_MIB:-$MIN_FREE_MIB}"
DEEPCORAL_MIN_FREE_MIB="${DEEPCORAL_MIN_FREE_MIB:-$MIN_FREE_MIB}"
CDAN_MIN_FREE_MIB="${CDAN_MIN_FREE_MIB:-$MIN_FREE_MIB}"
MAX_GPU_UTIL="${GPU_MAX_UTIL_PERCENT:-10}"
STABLE_CHECKS="${GPU_STABLE_CHECKS:-3}"
POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-0}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${RUN_ROOT:-$HERE/server_logs/dann_coral_cdan/$RUN_ID}"
BACKGROUND=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_server_dann_coral_cdan.sh [options]

Options:
  --dann-epochs N             DANN epochs (default 100)
  --deepcoral-epochs N        DeepCORAL epochs (default 20)
  --dann-iterations N         Exact DANN updates; 0 uses epochs
  --deepcoral-iterations N    Exact DeepCORAL updates; 0 uses epochs
  --cdan-iterations N         CDAN optimizer updates (default 2000)
  --targets "LIST"            Space/comma separated targets
  --seed N                    Random seed (default 42)
  --batch-size N              Per-domain image batch (default 16)
  --gpus LIST                 Physical GPU indices (default 0,1)
  --min-free-mib N            Minimum free VRAM for every method
  --dann-min-free-mib N       DANN-specific free VRAM threshold
  --deepcoral-min-free-mib N  DeepCORAL-specific threshold
  --cdan-min-free-mib N       CDAN-specific threshold
  --max-util-percent N        Maximum GPU utilization (default 10)
  --stable-checks N           Consecutive eligible polls (default 3)
  --poll-seconds N            Poll interval (default 30)
  --timeout-seconds N         GPU wait timeout; 0 waits forever
  --num-workers N             DataLoader workers (default 4)
  --prefetch-factor N         Batches prefetched per worker (default 4)
  --run-root PATH             Output/log directory
  --background                Start with nohup and print PID/log paths
  -h, --help                  Show this help

The three methods use exact-paired VIS+IR input, ResNet-50, final-epoch-only
target evaluation and no model checkpoint saving.
EOF
}

need_value() {
  if [ "$#" -lt 2 ]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

ORIGINAL_ARGS=("$@")
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dann-epochs) need_value "$@"; DANN_EPOCHS="$2"; shift 2 ;;
    --deepcoral-epochs) need_value "$@"; DEEPCORAL_EPOCHS="$2"; shift 2 ;;
    --dann-iterations) need_value "$@"; DANN_ITERATIONS="$2"; shift 2 ;;
    --deepcoral-iterations) need_value "$@"; DEEPCORAL_ITERATIONS="$2"; shift 2 ;;
    --cdan-iterations) need_value "$@"; CDAN_ITERATIONS="$2"; shift 2 ;;
    --targets) need_value "$@"; TARGET_TEXT="$2"; shift 2 ;;
    --seed) need_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --gpus) need_value "$@"; GPU_CANDIDATES="$2"; shift 2 ;;
    --min-free-mib)
      need_value "$@"; MIN_FREE_MIB="$2"
      DANN_MIN_FREE_MIB="$2"; DEEPCORAL_MIN_FREE_MIB="$2"; CDAN_MIN_FREE_MIB="$2"
      shift 2 ;;
    --dann-min-free-mib) need_value "$@"; DANN_MIN_FREE_MIB="$2"; shift 2 ;;
    --deepcoral-min-free-mib) need_value "$@"; DEEPCORAL_MIN_FREE_MIB="$2"; shift 2 ;;
    --cdan-min-free-mib) need_value "$@"; CDAN_MIN_FREE_MIB="$2"; shift 2 ;;
    --max-util-percent) need_value "$@"; MAX_GPU_UTIL="$2"; shift 2 ;;
    --stable-checks) need_value "$@"; STABLE_CHECKS="$2"; shift 2 ;;
    --poll-seconds) need_value "$@"; POLL_SECONDS="$2"; shift 2 ;;
    --timeout-seconds) need_value "$@"; WAIT_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --num-workers) need_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) need_value "$@"; PREFETCH_FACTOR="$2"; shift 2 ;;
    --run-root) need_value "$@"; RUN_ROOT="$2"; shift 2 ;;
    --background) BACKGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

positive_integer() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer: $value" >&2
    exit 2
  fi
}
nonnegative_number() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^([0-9]+)(\.[0-9]+)?$ ]]; then
    echo "$name must be non-negative: $value" >&2
    exit 2
  fi
}

positive_integer DANN_EPOCHS "$DANN_EPOCHS"
positive_integer DEEPCORAL_EPOCHS "$DEEPCORAL_EPOCHS"
positive_integer CDAN_ITERATIONS "$CDAN_ITERATIONS"
if ! [[ "$DANN_ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "DANN_ITERATIONS must be non-negative: $DANN_ITERATIONS" >&2; exit 2
fi
if ! [[ "$DEEPCORAL_ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "DEEPCORAL_ITERATIONS must be non-negative: $DEEPCORAL_ITERATIONS" >&2; exit 2
fi
positive_integer BATCH_SIZE "$BATCH_SIZE"
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS must be non-negative: $NUM_WORKERS" >&2; exit 2
fi
positive_integer PREFETCH_FACTOR "$PREFETCH_FACTOR"
positive_integer DANN_MIN_FREE_MIB "$DANN_MIN_FREE_MIB"
positive_integer DEEPCORAL_MIN_FREE_MIB "$DEEPCORAL_MIN_FREE_MIB"
positive_integer CDAN_MIN_FREE_MIB "$CDAN_MIN_FREE_MIB"
positive_integer STABLE_CHECKS "$STABLE_CHECKS"
nonnegative_number POLL_SECONDS "$POLL_SECONDS"
nonnegative_number WAIT_TIMEOUT_SECONDS "$WAIT_TIMEOUT_SECONDS"
if ! [[ "$MAX_GPU_UTIL" =~ ^[0-9]+$ ]] || (( MAX_GPU_UTIL < 0 || MAX_GPU_UTIL > 100 )); then
  echo "MAX_GPU_UTIL must be an integer in [0,100]: $MAX_GPU_UTIL" >&2; exit 2
fi
if [ "$POLL_SECONDS" = "0" ] || [ "$POLL_SECONDS" = "0.0" ]; then
  echo "POLL_SECONDS must be greater than zero" >&2; exit 2
fi
if [ "$GPU_CANDIDATES" != "0,1" ] && [ "$GPU_CANDIDATES" != "1,0" ] \
   && [ "$GPU_CANDIDATES" != "0" ] && [ "$GPU_CANDIDATES" != "1" ]; then
  echo "--gpus is restricted to physical GPU 0 and/or 1: $GPU_CANDIDATES" >&2
  exit 2
fi

if [ "$BACKGROUND" = "1" ]; then
  mkdir -p "$RUN_ROOT"
  OUT_LOG="$RUN_ROOT/launcher.out.log"
  ERR_LOG="$RUN_ROOT/launcher.err.log"
  PID_FILE="$RUN_ROOT/launcher.pid"
  FORWARD_ARGS=()
  for argument in "${ORIGINAL_ARGS[@]}"; do
    if [ "$argument" != "--background" ]; then FORWARD_ARGS+=("$argument"); fi
  done
  nohup bash "$HERE/scripts/run_server_dann_coral_cdan.sh" \
    "${FORWARD_ARGS[@]}" --run-root "$RUN_ROOT" \
    >"$OUT_LOG" 2>"$ERR_LOG" < /dev/null &
  CHILD_PID=$!
  echo "$CHILD_PID" > "$PID_FILE"
  echo "Background DANN/DeepCORAL/CDAN suite started."
  echo "PID=$CHILD_PID"
  echo "stdout=$OUT_LOG"
  echo "stderr=$ERR_LOG"
  echo "Monitor: tail -f '$OUT_LOG'"
  exit 0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Dataset root does not exist: $DATA_ROOT" >&2; exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available" >&2; exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2; exit 1
fi

export TORCH_HOME="${TORCH_HOME:-$HERE/pretrained_weights/torch}"
RESNET50_WEIGHT="$TORCH_HOME/hub/checkpoints/resnet50-0676ba61.pth"
if [ ! -f "$RESNET50_WEIGHT" ]; then
  echo "Missing ResNet-50 weight: $RESNET50_WEIGHT" >&2
  echo "Download it locally with tools/download_official_uda_weights.py and upload pretrained_weights." >&2
  exit 1
fi

TARGET_TEXT="${TARGET_TEXT//,/ }"
read -r -a TARGETS <<< "$TARGET_TEXT"
if [ "${#TARGETS[@]}" -eq 0 ]; then echo "No target was selected" >&2; exit 2; fi
for target in "${TARGETS[@]}"; do
  case "$target" in backlight|rainy|foggy|dark) ;; *)
    echo "Unknown target: $target" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/efficiency" "$RUN_ROOT/results" "$RUN_ROOT/prepared_exact_pairs"
LIST_DIR="$RUN_ROOT/prepared_exact_pairs"
echo "Preparing exact-paired VIS/IR lists: $LIST_DIR"
"$PYTHON_BIN" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT" --output-dir "$LIST_DIR" \
  --targets "${TARGETS[@]}" --modality paired || exit $?

echo "Checking physical GPUs $GPU_CANDIDATES before each job: free VRAM and GPU utilization."
echo "Budgets: DANN=$DANN_EPOCHS epochs/max_updates=$DANN_ITERATIONS; DeepCORAL=$DEEPCORAL_EPOCHS epochs/max_updates=$DEEPCORAL_ITERATIONS; CDAN=$CDAN_ITERATIONS iterations."
echo "Run root: $RUN_ROOT"

run_on_gpu() {
  local tag="$1" min_free="$2" log="$3" result_tag="$4" record_epochs="$5"
  local started finished elapsed status collect_status
  shift 5
  echo "[$(date --iso-8601=seconds)] queue $tag: free>=${min_free}MiB util<=${MAX_GPU_UTIL}% GPUs=$GPU_CANDIDATES" | tee -a "$log"
  started="$(date +%s)"
  "$PYTHON_BIN" "$HERE/tools/run_on_available_gpu.py" \
    --min-free-mib "$min_free" \
    --max-util-percent "$MAX_GPU_UTIL" \
    --stable-checks "$STABLE_CHECKS" \
    --poll-seconds "$POLL_SECONDS" \
    --timeout-seconds "$WAIT_TIMEOUT_SECONDS" \
    --gpus "$GPU_CANDIDATES" \
    --tag "$tag" \
    --lock-file /tmp/jmda_gpu_01_scheduler.lock \
    -- "$@" 2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  finished="$(date +%s)"
  elapsed=$((finished - started))
  collect_status="success"
  if [ "$status" -ne 0 ]; then
    collect_status="failed"
    echo "[FAILED] $tag exit=$status log=$log" >&2
  else
    echo "[OK] $tag" | tee -a "$log"
  fi
  if ! "$PYTHON_BIN" "$HERE/tools/collect_run_result.py" \
      --log "$log" --suite-id "$RUN_ID" --results-dir "$RUN_ROOT/results" \
      --protocol uda --run-tag "$result_tag" --status "$collect_status" \
      --exit-code "$status" --duration-seconds "$elapsed" \
      --epochs "$record_epochs" --seed "$SEED" \
      --efficiency-dir "$RUN_ROOT/efficiency" \
      --input-modalities VIS+IR --label-space-relation partial-set \
      --model-selection final_in_memory >> "$log" 2>&1; then
    echo "[FAILED] result collection for $tag" >&2
    return 1
  fi
  return "$status"
}

COMMON=(
  --seed "$SEED" --data-dir "$LIST_DIR"
  --efficiency_output_dir "$RUN_ROOT/efficiency"
  --save-dir "$RUN_ROOT/no_checkpoints"
  --num-workers "$NUM_WORKERS" --eval-num-workers 0
  --prefetch-factor "$PREFETCH_FACTOR" --worker-timeout-seconds 300
  --pin-memory --persistent-workers
  --multimodal --enhanced-augmentation --no-save-model
)
if [ "$NUM_WORKERS" = "0" ]; then
  COMMON=(
    --seed "$SEED" --data-dir "$LIST_DIR"
    --efficiency_output_dir "$RUN_ROOT/efficiency"
    --save-dir "$RUN_ROOT/no_checkpoints"
    --num-workers 0 --eval-num-workers 0
    --prefetch-factor "$PREFETCH_FACTOR" --worker-timeout-seconds 300
    --pin-memory --no-persistent-workers
    --multimodal --enhanced-augmentation --no-save-model
  )
fi

FAILURES=0
for target in "${TARGETS[@]}"; do
  tag="DANN_${target}_seed${SEED}"
  run_on_gpu "$tag" "$DANN_MIN_FREE_MIB" "$RUN_ROOT/logs/${tag}.log" \
    "dann_${target}" "$DANN_EPOCHS" \
    "$PYTHON_BIN" "$HERE/DANN/train_dann.py" --target "$target" \
    --epochs "$DANN_EPOCHS" --max-iterations "$DANN_ITERATIONS" \
    --backbone resnet50 --batch-size "$BATCH_SIZE" \
    "${COMMON[@]}" || FAILURES=$((FAILURES + 1))
done
for target in "${TARGETS[@]}"; do
  tag="DeepCORAL_${target}_seed${SEED}"
  run_on_gpu "$tag" "$DEEPCORAL_MIN_FREE_MIB" "$RUN_ROOT/logs/${tag}.log" \
    "deepcoral_${target}" "$DEEPCORAL_EPOCHS" \
    "$PYTHON_BIN" "$HERE/DeepCORAL/train_deepcoral.py" --target "$target" \
    --epochs "$DEEPCORAL_EPOCHS" --max-iterations "$DEEPCORAL_ITERATIONS" \
    --backbone resnet50 \
    --source-batch-size "$BATCH_SIZE" --target-batch-size "$BATCH_SIZE" \
    --test-batch-size "$BATCH_SIZE" "${COMMON[@]}" || FAILURES=$((FAILURES + 1))
done
for target in "${TARGETS[@]}"; do
  tag="CDAN_${target}_seed${SEED}"
  run_on_gpu "$tag" "$CDAN_MIN_FREE_MIB" "$RUN_ROOT/logs/${tag}.log" \
    "cdan_${target}" 0 \
    "$PYTHON_BIN" "$HERE/CDAN/train_cdan.py" --target "$target" \
    --num-iterations "$CDAN_ITERATIONS" --backbone ResNet50 \
    --batch-size "$BATCH_SIZE" "${COMMON[@]}" || FAILURES=$((FAILURES + 1))
done

"$PYTHON_BIN" -c "import sys; sys.path.insert(0, r'$HERE/tools'); from efficiency_common import generate_summary_files; generate_summary_files(r'$RUN_ROOT/efficiency')" \
  || echo "Warning: efficiency summary generation failed" >&2
echo "Suite finished: failures=$FAILURES, run_root=$RUN_ROOT"
if [ "$FAILURES" -ne 0 ]; then exit 1; fi
exit 0
