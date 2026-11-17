#!/usr/bin/env bash
# Server allocation: rainy UDA and separately reported label-guided dark.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHON_BIN="${PYTHON:-python}"
export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
export JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-0,1}"
export JMDA_GPU_MAX_UTIL="${JMDA_GPU_MAX_UTIL:-60}"
read -r -a SEEDS <<< "${WEATHER_ALL_SEEDS:-42 43 44 45 46}"
read -r -a RAINY_WEIGHTS <<< "${WEATHER_ALL_RAINY_WEIGHTS:-1 0.5}"
BACKLIGHT_LR="${WEATHER_STABILITY_BACKLIGHT_LR:-0.0001}"
LOG_DIR="$HERE/server_logs/weather_all_launcher"
mkdir -p "$LOG_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]]; then
  read -r ACTIVE_PID < "$LOG_DIR/latest.pid"
  if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    ACTIVE_CMD="$(ps -p "$ACTIVE_PID" -o args= || true)"
    case "$ACTIVE_CMD" in *run_weather_all.py*) echo "All-weather queue already running: PID=$ACTIVE_PID"; exit 1 ;; esac
  fi
fi
"$PYTHON_BIN" -B -u tools/run_weather_all.py --targets rainy dark --data-root "$JMDA_DATA_ROOT" --check-environment
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$HERE/runs/weather_all/server_${STAMP}"
LOG="$LOG_DIR/${STAMP}.log"
nohup "$PYTHON_BIN" -B -u tools/run_weather_all.py --seeds "${SEEDS[@]}" \
  --targets rainy dark \
  --rainy-weights "${RAINY_WEIGHTS[@]}" --backlight-image-lr "$BACKLIGHT_LR" \
  --data-root "$JMDA_DATA_ROOT" --output "$RUN_DIR" --execute > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$LOG_DIR/${STAMP}.pid"
printf '%s\n' "$PID" > "$LOG_DIR/latest.pid"
printf '%s\n' "$LOG" > "$LOG_DIR/latest.log"
TOTAL=$(( ${#SEEDS[@]} * (5 * ${#RAINY_WEIGHTS[@]} + 5) ))
echo "Server: rainy/dark only; all five methods. Seeds=${SEEDS[*]}. Backlight/foggy run locally."
echo "Protocols: rainy=UDA; dark=LABEL_GUIDED_CLASS_ALIGNMENT. Report separately."
echo "Total=$TOTAL (default 75); rainy weights=${RAINY_WEIGHTS[*]}; no source-only controls in this new queue."
echo "PID=$PID"
echo "Run directory=$RUN_DIR"
echo "Metrics=$RUN_DIR/summary.csv; raw metrics=$RUN_DIR/runs.csv; selected diagnostics=$RUN_DIR/diagnostics.jsonl"
echo "log=$LOG"
echo "Monitor: tail -f '$LOG'"
