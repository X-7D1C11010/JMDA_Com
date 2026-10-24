#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}"; exit 1 ;; esac
LOG_DIR="$HERE/server_logs/vds2raw_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/${MODE}_${STAMP}.log"
PID_FILE="$LOG_DIR/${MODE}_${STAMP}.pid"
nohup bash "$HERE/scripts/run_vds2raw_methods.sh" "$MODE" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "PID=$PID"
echo "log=$LOG"
echo "tail -f '$LOG'"
