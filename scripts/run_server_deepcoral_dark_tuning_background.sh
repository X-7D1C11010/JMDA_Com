#!/usr/bin/env bash
# Latest task defaults to ONE dark Deep CORAL test: adaptation LR scale 1, seed46.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHON_BIN="${PYTHON:-python}"
export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
export JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-0,1}"
export JMDA_GPU_MAX_UTIL="${JMDA_GPU_MAX_UTIL:-60}"
export JMDA_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TRIALS=("$@")
if (( ${#TRIALS[@]} == 0 )); then TRIALS=(phase_lr1); fi
LOG_DIR="$HERE/server_logs/deepcoral_dark_tuning_launcher"
mkdir -p "$LOG_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]]; then
  read -r ACTIVE_PID < "$LOG_DIR/latest.pid"
  if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    ACTIVE_CMD="$(ps -p "$ACTIVE_PID" -o args= || true)"
    case "$ACTIVE_CMD" in *run_deepcoral_dark_tuning.py*) echo "Tuning queue already running: PID=$ACTIVE_PID"; exit 1 ;; esac
  fi
fi
"$PYTHON_BIN" -B -u tools/run_deepcoral_dark_tuning.py --trials "${TRIALS[@]}" --check-environment
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$HERE/runs/weather_label_guided/server_fewshot_deepcoral_dark_tuning_${STAMP}"
LOG="$LOG_DIR/${STAMP}.log"
nohup "$PYTHON_BIN" -B -u tools/run_deepcoral_dark_tuning.py --trials "${TRIALS[@]}" --output "$RUN_DIR" --execute > "$LOG" 2>&1 < /dev/null &
WORKER_PID=$!
printf '%s\n' "$WORKER_PID" > "$LOG_DIR/${STAMP}.pid"
printf '%s\n' "$WORKER_PID" > "$LOG_DIR/latest.pid"
printf '%s\n' "$LOG" > "$LOG_DIR/latest.log"
printf '%s\n' "$RUN_DIR" > "$LOG_DIR/latest.run.txt"
echo "Dark Deep CORAL only: ${#TRIALS[@]} test(s); seed46; 40 epochs each."
echo "Trials=${TRIALS[*]}"
echo "Same 3 labels/class, class pool and split; no target classification CE."
echo "GPU selection checks >=6144MiB free and <=${JMDA_GPU_MAX_UTIL}% utilization."
echo "PID=$WORKER_PID"
echo "Run directory=$RUN_DIR"
echo "log=$LOG"
echo "Monitor: tail -f '$LOG'"
