#!/usr/bin/env bash
# One-command nohup wrapper for TIR followed by rainy/dark MCFT.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIS_INITIAL_SCALE="${1:-${VDS2RAW_TIR_AIS_INITIAL_SCALE:-1.50}}"
MCFT_EPOCHS="${2:-${MCFT_RAINY_DARK_EPOCHS:-100}}"
TIR_EPOCHS="${3:-${VDS2RAW_TIR_EPOCHS:-100}}"

LOG_DIR="$HERE/server_logs/tir_mcft_rainy_dark_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/formal_${STAMP}.log"
PID_FILE="$LOG_DIR/formal_${STAMP}.pid"

nohup bash "$HERE/scripts/run_tir_mcft_rainy_dark_sequential.sh" \
  "$AIS_INITIAL_SCALE" "$MCFT_EPOCHS" "$TIR_EPOCHS" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"

echo "PID=$PID"
echo "log=$LOG"
echo "pid_file=$PID_FILE"
echo "tail -f '$LOG'"
