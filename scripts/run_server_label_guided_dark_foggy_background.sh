#!/usr/bin/env bash
# Independent supervised class-matching/alignment; never report this as UDA.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHON_BIN="${PYTHON:-python}"
export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
export JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-0,1}"
# Explicitly record/reproduce numerical settings for the formal server run.
export JMDA_DETERMINISTIC="${JMDA_DETERMINISTIC:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
read -r -a SEEDS <<< "${JMDA_LABEL_GUIDED_SEEDS:-42 43 44 45 46}"
PARAMETER_PROFILE="${JMDA_LABEL_GUIDED_PROFILE:-reference}"
EVALUATION_MODE="${JMDA_LABEL_GUIDED_EVALUATION_MODE:-validation_ce}"
case "$PARAMETER_PROFILE" in reference|conservative|lr_only) ;; *) echo "Invalid JMDA_LABEL_GUIDED_PROFILE" >&2; exit 2 ;; esac
case "$EVALUATION_MODE" in validation_peak|validation_ce|holdout_ce) ;; *) echo "Invalid JMDA_LABEL_GUIDED_EVALUATION_MODE" >&2; exit 2 ;; esac
LOG_DIR="$HERE/server_logs/label_guided_dark_foggy_launcher"
mkdir -p "$LOG_DIR"
if [[ -f "$LOG_DIR/latest.pid" ]]; then
  read -r ACTIVE_PID < "$LOG_DIR/latest.pid"
  if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    ACTIVE_CMD="$(ps -p "$ACTIVE_PID" -o args= || true)"
    case "$ACTIVE_CMD" in
      *run_label_guided_matrix.py*) echo "Label-guided queue already running: PID=$ACTIVE_PID"; exit 1 ;;
    esac
  fi
fi
"$PYTHON_BIN" -B -c "import torch, timm; assert torch.cuda.is_available(), 'CUDA is not available'"
"$PYTHON_BIN" -B -u tools/download_official_uda_weights.py --check-only \
  --only resnet18 convnext_small unip_tiny unip_small
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
BASE_LIST_DIR="$HERE/prepared_lists/label_guided_base_${STAMP}"
"$PYTHON_BIN" -B -u tools/prepare_paired_protocol_lists.py \
  --data-root "$JMDA_DATA_ROOT" --output-dir "$BASE_LIST_DIR" --targets dark foggy --modality paired
LIST_DIR="$HERE/prepared_lists/label_guided_holdout_${STAMP}"
"$PYTHON_BIN" -B -u tools/prepare_weather_generalization_split.py \
  --input-dir "$BASE_LIST_DIR" --output-dir "$LIST_DIR" --targets dark foggy \
  --test-fraction 0.2 --split-seed 20261115
RUN_DIR="$HERE/runs/weather_label_guided/server_${STAMP}"
LOG="$LOG_DIR/${STAMP}.log"
nohup "$PYTHON_BIN" -B -u tools/run_label_guided_matrix.py \
  --data-dir "$LIST_DIR" --seeds "${SEEDS[@]}" --gpus "$JMDA_GPU_CANDIDATES" \
  --output "$RUN_DIR" --complete-budget --evaluation-mode "$EVALUATION_MODE" \
  --parameter-profile "$PARAMETER_PROFILE" --execute > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$LOG_DIR/${STAMP}.pid"
printf '%s\n' "$PID" > "$LOG_DIR/latest.pid"
echo "Protocol=LABEL_GUIDED_CLASS_ALIGNMENT; target training labels USED. This is not UDA."
echo "DANN/DeepCORAL/CDAN/MDD/MCC x dark/foggy; seeds=${SEEDS[*]}; sequential GPU queue."
echo "Parameter profile=$PARAMETER_PROFILE; evaluation mode=$EVALUATION_MODE."
echo "Default tuning uses validation CE only; holdout_ce is for a separately frozen final configuration."
echo "Holdout groups are exact VIS/IR content pairs, not verified scene/MMSI groups; historical runs used the complete train set."
echo "PID=$PID"
echo "results=$RUN_DIR"
echo "log=$LOG"
echo "Monitor: tail -f '$LOG'"
