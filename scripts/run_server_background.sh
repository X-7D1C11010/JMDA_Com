#!/usr/bin/env bash
# Run the server experiment under nohup so it survives an SSH disconnect.
#
# Usage:
#   bash scripts/run_server_background.sh uda
#   bash scripts/run_server_background.sh jmda_uda 100
#
# All exported configuration variables (JMDA_DATA_ROOT,
# JMDA_CLASS_NAMES_FILE, CUDA_VISIBLE_DEVICES, etc.) are inherited.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -lt 1 ]; then
  echo "用法: bash scripts/run_server_background.sh {sda|jmda_uda|jmda_oracle_shared_uda|uda|supervised} [epochs]"
  exit 1
fi
if ! command -v nohup >/dev/null 2>&1; then
  echo "找不到 nohup，无法保证断开 SSH 后任务继续运行。"
  exit 1
fi

PROTOCOL="$1"
case "$PROTOCOL" in
  sda|jmda_uda|jmda_oracle_shared_uda|uda|supervised) ;;
  *)
    echo "未知协议: $PROTOCOL"
    exit 1
    ;;
esac

if [ "$PROTOCOL" = "uda" ]; then
  CLASS_NAMES_FILE="${JMDA_CLASS_NAMES_FILE:-}"
  PYTHON_BIN="${PYTHON:-python}"
  if [ -z "$CLASS_NAMES_FILE" ] || [ ! -f "$CLASS_NAMES_FILE" ]; then
    echo "无法启动：JMDA_CLASS_NAMES_FILE 必须指向真实存在的 14 类英文名称文件。"
    echo "注意：/path/to/14_class_names.txt 只是示例占位路径。"
    exit 1
  fi
  if ! "$PYTHON_BIN" "$HERE/tools/class_names.py" \
      "$CLASS_NAMES_FILE" --expected-count 14; then
    echo "无法启动：类别名称文件格式或内容无效。"
    exit 1
  fi
fi

LAUNCH_LOG_DIR="$HERE/server_logs/launcher"
mkdir -p "$LAUNCH_LOG_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LAUNCH_LOG="$LAUNCH_LOG_DIR/${PROTOCOL}_${RUN_STAMP}.log"
PID_FILE="$LAUNCH_LOG_DIR/${PROTOCOL}_${RUN_STAMP}.pid"

nohup bash "$HERE/scripts/run_server.sh" "$@" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$PID_FILE"

echo "后台任务已启动。"
echo "PID=$TRAIN_PID"
echo "启动日志=$LAUNCH_LOG"
echo "PID文件=$PID_FILE"
echo "查看日志: tail -f '$LAUNCH_LOG'"
echo "检查进程: ps -p $TRAIN_PID -o pid,etime,cmd"
