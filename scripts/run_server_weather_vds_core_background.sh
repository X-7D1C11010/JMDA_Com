#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $SEED" >&2; exit 1; }
LOG_DIR="$HERE/server_logs/weather_vds_core_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/seed${SEED}_${STAMP}.log"
PID_FILE="$LOG_DIR/seed${SEED}_${STAMP}.pid"
nohup bash "$HERE/scripts/run_weather_vds_core_methods.sh" "$SEED" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "PID=$PID"
echo "log=$LOG"
echo "PID file=$PID_FILE"
echo "Monitor: tail -f '$LOG'"
