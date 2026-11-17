#!/usr/bin/env bash
# Compatibility entry: default delegates to rainy/dark server allocation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHON_BIN="${PYTHON:-python}"
SCOPE="${1:-all}"
if [[ "$SCOPE" == all || "$SCOPE" == both ]]; then exec bash "$HERE/scripts/run_server_weather_all_background.sh"; fi
if [[ "$SCOPE" == backlight ]]; then echo "Backlight now runs locally: scripts/run_local_weather_split.ps1 -Background" >&2; exit 2; fi
case "$SCOPE" in rainy) ;; *) echo "usage: $0 {all|both|rainy}" >&2; exit 2 ;; esac
export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
export JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-0,1}"
export JMDA_GPU_MAX_UTIL="${JMDA_GPU_MAX_UTIL:-60}"
read -r -a BACKLIGHT_SEEDS <<< "${WEATHER_STABILITY_BACKLIGHT_SEEDS:-42 43 44 45 46}"
read -r -a RAINY_SEEDS <<< "${WEATHER_STABILITY_RAINY_SEEDS:-42 43 44 45 46}"
BACKLIGHT_LR="${WEATHER_STABILITY_BACKLIGHT_LR:-0.0001}"
LOG_DIR="$HERE/server_logs/weather_stability_launcher"
mkdir -p "$LOG_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]]; then
  read -r ACTIVE_PID < "$LOG_DIR/latest.pid"
  if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    ACTIVE_CMD="$(ps -p "$ACTIVE_PID" -o args= || true)"
    case "$ACTIVE_CMD" in *run_weather_stability.py*) echo "Stability queue already running: PID=$ACTIVE_PID"; exit 1 ;; esac
  fi
fi
"$PYTHON_BIN" -B -u tools/run_weather_stability.py --scope "$SCOPE" \
  --backlight-seeds "${BACKLIGHT_SEEDS[@]}" --backlight-methods cdan mcc --rainy-seeds "${RAINY_SEEDS[@]}" \
  --backlight-image-lr "$BACKLIGHT_LR" --data-root "$JMDA_DATA_ROOT" --check-environment
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$LOG_DIR/${STAMP}.log"
RUN_DIR="$HERE/runs/weather_stability/server_${SCOPE}_${STAMP}"
nohup "$PYTHON_BIN" -B -u tools/run_weather_stability.py --scope "$SCOPE" \
  --backlight-seeds "${BACKLIGHT_SEEDS[@]}" --backlight-methods cdan mcc --rainy-seeds "${RAINY_SEEDS[@]}" \
  --backlight-image-lr "$BACKLIGHT_LR" --output "$RUN_DIR" \
  --data-root "$JMDA_DATA_ROOT" --execute > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$LOG_DIR/${STAMP}.pid"
printf '%s\n' "$PID" > "$LOG_DIR/latest.pid"
printf '%s\n' "$LOG" > "$LOG_DIR/latest.log"
REQUESTED=0
if [[ "$SCOPE" != rainy ]]; then REQUESTED=$((REQUESTED + 2 * ${#BACKLIGHT_SEEDS[@]})); fi
if [[ "$SCOPE" != backlight ]]; then REQUESTED=$((REQUESTED + 10 * ${#RAINY_SEEDS[@]})); fi
echo "UDA: backlight CDAN/MCC only; rainy five methods at distillation weights 1/0.5."
echo "Scope=$SCOPE; backlight seeds=${BACKLIGHT_SEEDS[*]}; rainy seeds=${RAINY_SEEDS[*]}; backlight encoder LR=$BACKLIGHT_LR"
echo "Requested=$REQUESTED; this explicit subset is rainy only; omit scope or use all for rainy/dark server allocation."
echo "Preflight checks actual encoder gradients and updates; target instance labels are not used."
echo "PID=$PID"
echo "Run directory=$RUN_DIR"
echo "Metrics=$RUN_DIR/summary.csv; per-run metrics=$RUN_DIR/runs.csv; progress=$RUN_DIR/queue_status.json"
echo "log=$LOG"
echo "Monitor: tail -f '$LOG'"
