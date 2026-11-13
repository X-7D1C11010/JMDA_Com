#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $SEED" >&2; exit 1; }
UPDATE="$HERE/experiment_updates/mcc_weather_v2_20261113"
if [ ! -f "$UPDATE/scripts/run_pending_adaptation_v2.sh" ]; then
  echo "Extract mcc_weather_v2_update.zip into the project root first; missing: $UPDATE" >&2
  exit 1
fi
export JMDA_PROJECT_ROOT="$HERE"
LOG_DIR="$HERE/server_logs/pending_adaptation_v2_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/seed${SEED}_${STAMP}.log"
PID_FILE="$LOG_DIR/seed${SEED}_${STAMP}.pid"
nohup bash "$UPDATE/scripts/run_pending_adaptation_v2.sh" "$SEED" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "PID=$PID"
echo "Isolated code=$UPDATE"
echo "log=$LOG"
echo "PID file=$PID_FILE"
echo "Monitor: tail -f '$LOG'"
