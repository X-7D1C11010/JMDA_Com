#!/usr/bin/env bash
# Independent candidate: dark five methods + foggy MDD/MCC. Default pilot seed46.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHON_BIN="${PYTHON:-python}"
export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
export JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-0,1}"
export JMDA_GPU_MAX_UTIL="${JMDA_GPU_MAX_UTIL:-60}"
export JMDA_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
read -r -a SEEDS <<< "${*:-46}"
LOG_DIR="$HERE/server_logs/weather_fewshot_v2_server_launcher"
mkdir -p "$LOG_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]]; then
  read -r ACTIVE_PID < "$LOG_DIR/latest.pid"
  if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    ACTIVE_CMD="$(ps -p "$ACTIVE_PID" -o args= || true)"
    case "$ACTIVE_CMD" in *run_weather_fewshot_suite.py*) echo "Adaptive queue already running: PID=$ACTIVE_PID"; exit 1 ;; esac
  fi
fi
"$PYTHON_BIN" -B -u tools/run_weather_fewshot_suite.py --machine server --fewshot-recipe server_adaptive_v2 \
  --seeds "${SEEDS[@]}" --data-root "$JMDA_DATA_ROOT" --check-environment
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$HERE/runs/weather_label_guided/server_fewshot_v2_${STAMP}"
LOG="$LOG_DIR/${STAMP}.log"
nohup "$PYTHON_BIN" -B -u tools/run_weather_fewshot_suite.py --machine server --fewshot-recipe server_adaptive_v2 \
  --seeds "${SEEDS[@]}" --data-root "$JMDA_DATA_ROOT" --output "$RUN_DIR" --execute > "$LOG" 2>&1 < /dev/null &
WORKER_PID=$!
printf '%s\n' "$WORKER_PID" > "$LOG_DIR/${STAMP}.pid"
printf '%s\n' "$WORKER_PID" > "$LOG_DIR/latest.pid"
printf '%s\n' "$LOG" > "$LOG_DIR/latest.log"
printf '%s\n' "$RUN_DIR" > "$LOG_DIR/latest.run.txt"
echo "Server candidate: dark DANN/DeepCORAL/CDAN/MDD/MCC; foggy MDD/MCC."
echo "Seeds=${SEEDS[*]}; jobs=$((7 * ${#SEEDS[@]})); 40 epochs; 3 labels/class; alignment=0.2; source EMA=0.9."
echo "GPU admission: free>=6144MiB, utilization<=${JMDA_GPU_MAX_UTIL}%, 3 consecutive checks."
echo "Peak ACC and same-epoch P/R/F1 are reported; CE-selected results are separate."
echo "PID=$WORKER_PID"
echo "Run directory=$RUN_DIR"
echo "log=$LOG"
echo "Monitor: tail -f '$LOG'"
