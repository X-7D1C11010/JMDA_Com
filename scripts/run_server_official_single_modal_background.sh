#!/usr/bin/env bash
# Start the official single-modal suite detached from the SSH session.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-backlight}"
SEED="${2:-42}"
MODALITIES="${3:-vis,ir}"
METHODS="${4:-DANN,DeepCORAL,CDAN,MDD,MCC}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LAUNCH_DIR="$HERE/server_logs/official_single_modal_launcher"
mkdir -p "$LAUNCH_DIR"
OUT_LOG="$LAUNCH_DIR/${TARGET}_seed${SEED}_$STAMP.out.log"
ERR_LOG="$LAUNCH_DIR/${TARGET}_seed${SEED}_$STAMP.err.log"
PID_FILE="$LAUNCH_DIR/${TARGET}_seed${SEED}_$STAMP.pid"

nohup bash "$HERE/scripts/run_server_official_single_modal.sh" \
  "$TARGET" "$SEED" "$MODALITIES" "$METHODS" \
  >"$OUT_LOG" 2>"$ERR_LOG" < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"

echo "Official single-modal background run started."
echo "PID=$pid"
echo "stdout=$OUT_LOG"
echo "stderr=$ERR_LOG"
echo "Monitor: tail -f '$OUT_LOG'"
