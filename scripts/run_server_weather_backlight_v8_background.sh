#!/usr/bin/env bash
# Only the backlight presets that passed the local seed42 prefix screen.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $SEED" >&2; exit 1; }
PYTHON_BIN="${PYTHON:-python}"
UPDATE="$("$PYTHON_BIN" -B "$HERE/tools/prepare_mcc_weather_v2_snapshot.py" --project-root "$HERE")"
LOG_DIR="$HERE/server_logs/weather_qualified_v8_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/backlight_seed${SEED}_${STAMP}.log"
PID_FILE="$LOG_DIR/backlight_seed${SEED}_${STAMP}.pid"
nohup "$PYTHON_BIN" -B -u "$UPDATE/weather/run_qualified_backlight.py" \
  --project-root "$HERE" --seed "$SEED" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "PID=$PID"
echo "Isolated code=$UPDATE"
echo "Backlight only; DANN -> Deep CORAL -> CDAN -> MDD -> MCC; 60 epochs; batch/domain=16"
echo "Each method waits for sufficient GPU memory AND spare utilization."
echo "log=$LOG"
echo "PID file=$PID_FILE"
echo "Monitor: tail -f '$LOG'"
