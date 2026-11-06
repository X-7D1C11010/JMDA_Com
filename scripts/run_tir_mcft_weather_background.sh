#!/usr/bin/env bash
# One-command nohup launcher for run_tir_mcft_weather_sequential.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in
  formal|smoke) ;;
  *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;;
esac

LOG_DIR="$HERE/server_logs/tir_mcft_weather_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/${MODE}_${STAMP}.log"
PID_FILE="$LOG_DIR/${MODE}_${STAMP}.pid"

nohup bash "$HERE/scripts/run_tir_mcft_weather_sequential.sh" "$MODE" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"

echo "PID=$PID"
echo "log=$LOG"
echo "pid_file=$PID_FILE"
echo "tail -f '$LOG'"

