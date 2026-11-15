#!/usr/bin/env bash
# Default: missing backlight seeds42/43 + rainy seeds42--46, all five methods.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOPE="${1:-both}"
case "$SCOPE" in
  both|backlight|rainy) ;;
  *) echo "usage: $0 {both|backlight|rainy}" >&2; exit 2 ;;
esac
PYTHON_BIN="${PYTHON:-python}"
UPDATE="$("$PYTHON_BIN" -B "$HERE/tools/prepare_mcc_weather_v2_snapshot.py" --project-root "$HERE")"
"$PYTHON_BIN" -B -u "$UPDATE/tools/run_backlight_rainy_suite.py" \
  --project-root "$HERE" --scope "$SCOPE" --check-environment
LOG_DIR="$HERE/server_logs/backlight_rainy_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/${SCOPE}_${STAMP}.log"
PID_FILE="$LOG_DIR/${SCOPE}_${STAMP}.pid"
nohup "$PYTHON_BIN" -B -u "$UPDATE/tools/run_backlight_rainy_suite.py" \
  --project-root "$HERE" --scope "$SCOPE" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "PID=$PID; scope=$SCOPE"
echo "Backlight repair: seeds42/43 only; rainy: seeds42--46; five methods; sequential GPU queue."
echo "log=$LOG"
echo "PID file=$PID_FILE"
echo "Monitor: tail -f '$LOG'"
