#!/usr/bin/env bash
# One-command nohup launcher for the rainy+dark MCFT five-seed experiment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${1:-${MCFT_RAINY_DARK_EPOCHS:-100}}"
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive_epochs]" >&2
  exit 1
fi

LOG_DIR="$HERE/server_logs/mcft_rainy_dark_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/epochs${EPOCHS}_${STAMP}.log"
PID_FILE="$LOG_DIR/epochs${EPOCHS}_${STAMP}.pid"

nohup bash "$HERE/scripts/run_mcft_rainy_dark_sequential.sh" "$EPOCHS" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"

echo "PID=$PID"
echo "log=$LOG"
echo "pid_file=$PID_FILE"
echo "tail -f '$LOG'"
