#!/usr/bin/env bash
# VDS MCC/CDAN x5 and qualified backlight five methods x5; 35 runs by default.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOPE="${1:-both}"
case "$SCOPE" in
  both) TASKS=35 ;;
  vds) TASKS=10 ;;
  backlight) TASKS=25 ;;
  *) echo "usage: $0 {both|vds|backlight}" >&2; exit 1 ;;
esac
PYTHON_BIN="${PYTHON:-python}"
UPDATE="$("$PYTHON_BIN" -B "$HERE/tools/prepare_mcc_weather_v2_snapshot.py" --project-root "$HERE")"
LOG_DIR="$HERE/server_logs/vds_backlight_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/${SCOPE}_${STAMP}.log"
PID_FILE="$LOG_DIR/${SCOPE}_${STAMP}.pid"
nohup "$PYTHON_BIN" -B -u "$UPDATE/tools/run_vds_backlight_suite.py" \
  --project-root "$HERE" --scope "$SCOPE" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "PID=$PID"
echo "Scope=$SCOPE; requested=$TASKS; seeds42-46; VDS MCC/CDAN and/or backlight five methods"
echo "Isolated code=$UPDATE"
echo "log=$LOG"
echo "PID file=$PID_FILE"
echo "Monitor: tail -f '$LOG'"
