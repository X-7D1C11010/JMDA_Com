#!/usr/bin/env bash
# Start the sequential CCSA -> d-SNE -> MCFT suite with nohup.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in
  formal|smoke) ;;
  *)
    echo "用法: bash scripts/run_ccsa_dsne_mcft_background.sh {formal|smoke}"
    exit 1
    ;;
esac
if ! command -v nohup >/dev/null 2>&1; then
  echo "找不到 nohup，无法保证断开 SSH 后任务继续运行。"
  exit 1
fi

LAUNCH_LOG_DIR="$HERE/server_logs/ccsa_dsne_mcft_launcher"
mkdir -p "$LAUNCH_LOG_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LAUNCH_LOG="$LAUNCH_LOG_DIR/${MODE}_${RUN_STAMP}.log"
PID_FILE="$LAUNCH_LOG_DIR/${MODE}_${RUN_STAMP}.pid"

nohup bash "$HERE/scripts/run_ccsa_dsne_mcft.sh" "$MODE" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$PID_FILE"

echo "CCSA -> d-SNE -> MCFT 后台任务已启动。"
echo "PID=$TRAIN_PID"
echo "启动日志=$LAUNCH_LOG"
echo "PID文件=$PID_FILE"
echo "查看日志: tail -f '$LAUNCH_LOG'"
echo "检查进程: ps -p $TRAIN_PID -o pid,etime,cmd"
