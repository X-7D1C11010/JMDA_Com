#!/usr/bin/env bash
# Independent reduced-strength label-guided dark experiment; no rainy jobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHON_BIN="${PYTHON:-python}"
export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
export JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-0,1}"
export JMDA_GPU_MAX_UTIL="${JMDA_GPU_MAX_UTIL:-60}"
export JMDA_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
read -r -a SEEDS <<< "${*:-${JMDA_DARK_ALIGNMENT_SEEDS:-42}}"
read -r -a METHODS <<< "${JMDA_DARK_ALIGNMENT_METHODS:-dann deepcoral cdan mdd mcc}"
PAIRS=()
for seed in "${SEEDS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seeds must be nonnegative integers" >&2; exit 2; }
done
for method in "${METHODS[@]}"; do
  case "$method" in dann|deepcoral|cdan|mdd|mcc) PAIRS+=("${method}_dark") ;; *) echo "Invalid dark method: $method" >&2; exit 2 ;; esac
done
[[ ${#PAIRS[@]} -gt 0 && ${#SEEDS[@]} -gt 0 ]] || { echo "Empty method/seed list" >&2; exit 2; }
LOG_DIR="$HERE/server_logs/dark_alignment_launcher"
mkdir -p "$LOG_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]]; then
  read -r ACTIVE_PID < "$LOG_DIR/latest.pid"
  if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    ACTIVE_CMD="$(ps -p "$ACTIVE_PID" -o args= || true)"
    case "$ACTIVE_CMD" in *run_label_guided_matrix.py*) echo "Dark alignment queue already running: PID=$ACTIVE_PID"; exit 1 ;; esac
  fi
fi
"$PYTHON_BIN" -B -u tools/run_weather_all.py --targets dark --data-root "$JMDA_DATA_ROOT" --check-environment
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_LIST_DIR="$HERE/prepared_lists/dark_alignment_base_${STAMP}"
LIST_DIR="$HERE/prepared_lists/dark_alignment_holdout_${STAMP}"
"$PYTHON_BIN" -B -u tools/prepare_paired_protocol_lists.py \
  --data-root "$JMDA_DATA_ROOT" --output-dir "$BASE_LIST_DIR" --targets dark --modality paired
"$PYTHON_BIN" -B -u tools/prepare_weather_generalization_split.py \
  --input-dir "$BASE_LIST_DIR" --output-dir "$LIST_DIR" --targets dark --test-fraction .2 --split-seed 20261115
RUN_DIR="$HERE/runs/weather_label_guided/server_dark_alignment1_${STAMP}"
LOG="$LOG_DIR/${STAMP}.log"
nohup "$PYTHON_BIN" -B -u tools/run_label_guided_matrix.py \
  --data-dir "$LIST_DIR" --pairs "${PAIRS[@]}" --seeds "${SEEDS[@]}" \
  --gpus "$JMDA_GPU_CANDIDATES" --matched-alignment-weight 1 \
  --parameter-profile reference --complete-budget --evaluation-mode validation_ce \
  --output "$RUN_DIR" --execute > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$LOG_DIR/${STAMP}.pid"
printf '%s\n' "$PID" > "$LOG_DIR/latest.pid"
printf '%s\n' "$LOG" > "$LOG_DIR/latest.log"
printf '%s\n' "$RUN_DIR" > "$LOG_DIR/latest.run.txt"
echo "Dark only: methods=${METHODS[*]}; seeds=${SEEDS[*]}; jobs=$(( ${#METHODS[@]} * ${#SEEDS[@]} ))."
echo "Alignment coefficient 5 -> 1; other reference training settings unchanged."
echo "Protocol=LABEL_GUIDED_CLASS_ALIGNMENT; target training truth still guides pairs/alignment. Not UDA."
echo "Validation CE selection; no heldout evaluation. GPU queue checks free memory and utilization."
echo "PID=$PID"
echo "Run directory=$RUN_DIR"
echo "Metrics=$RUN_DIR/summary_by_pair.csv; per-run metrics=$RUN_DIR/real_results_inventory.json"
echo "log=$LOG"
echo "Monitor: tail -f '$LOG'"
