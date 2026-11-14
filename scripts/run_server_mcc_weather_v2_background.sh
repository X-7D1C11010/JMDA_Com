#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-42}"
SCOPE="${2:-vds-only}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $SEED" >&2; exit 1; }
case "$SCOPE" in all|cdan-only|v3|v4|vds-only) ;; *) echo "usage: $0 SEED {all|cdan-only|v3|v4|vds-only}" >&2; exit 1 ;; esac
if [ "$SCOPE" = v4 ]; then
  echo "Weather balanced_v4 did not pass the local seed42 screen; the combined v4 queue is not released." >&2
  echo "Use: bash scripts/run_server_mcc_weather_v2_background.sh $SEED vds-only" >&2
  echo "No background task was started." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON:-python}"
PREPARE="$HERE/tools/prepare_mcc_weather_v2_snapshot.py"
if [ ! -f "$PREPARE" ]; then
  echo "Git checkout is missing $PREPARE; synchronize the latest code first." >&2
  exit 1
fi
UPDATE="$("$PYTHON_BIN" -B "$PREPARE" --project-root "$HERE")"
export JMDA_PROJECT_ROOT="$HERE"
LOG_DIR="$HERE/server_logs/pending_adaptation_v2_launcher"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/${SCOPE}_seed${SEED}_${STAMP}.log"
PID_FILE="$LOG_DIR/${SCOPE}_seed${SEED}_${STAMP}.pid"
nohup bash "$UPDATE/scripts/run_pending_adaptation_v2.sh" "$SEED" "$SCOPE" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "PID=$PID"
echo "Isolated code=$UPDATE"
echo "Scope=$SCOPE (vds-only: heading/size MCC x5 -> CDAN x5, seeds 42-46; weather v4 withheld after local screening; v3/all/cdan-only: historical configurations)"
echo "log=$LOG"
echo "PID file=$PID_FILE"
echo "Monitor: tail -f '$LOG'"
