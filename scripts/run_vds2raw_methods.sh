#!/usr/bin/env bash
# Sequential VDS2Raw CCSA -> d-SNE -> three-modal MCFT runner.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}"; exit 1 ;; esac

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${VDS2RAW_DATA_ROOT:-/home/lixiang/lx/VDS2Raw/threeclass_ready}"
OUTPUT_ROOT="${VDS2RAW_OUTPUT_ROOT:-$HERE/runs/vds2raw}"
read -r -a SEEDS <<< "${VDS2RAW_SEEDS:-42}"
CCSA_EPOCHS="${VDS2RAW_CCSA_EPOCHS:-80}"
DSNE_EPOCHS="${VDS2RAW_DSNE_EPOCHS:-20}"
MCFT_EPOCHS="${VDS2RAW_MCFT_EPOCHS:-500}"
CCSA_BATCH="${VDS2RAW_CCSA_BATCH_SIZE:-256}"
DSNE_BATCH="${VDS2RAW_DSNE_BATCH_SIZE:-32}"
MCFT_BATCH="${VDS2RAW_MCFT_BATCH_SIZE:-64}"
DSNE_LAYERS="${VDS2RAW_DSNE_LAYERS:-101}"
MCFT_EVAL="${VDS2RAW_MCFT_EVAL_FREQUENCY:-10}"
MAX_BATCH_ARGS=()
if [ "$MODE" = smoke ]; then
  CCSA_EPOCHS=1; DSNE_EPOCHS=1; MCFT_EPOCHS=1
  CCSA_BATCH=8; DSNE_BATCH=4; MCFT_BATCH=4; DSNE_LAYERS=18; MCFT_EVAL=1
  MAX_BATCH_ARGS=(--max-train-batches 1)
fi

RUN_ID="vds2raw_${MODE}_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
ARTIFACT_DIR="$RUN_DIR/artifacts"
LOG_DIR="$RUN_DIR/logs"
SUMMARY_DIR="$RUN_DIR/summary"
STATS_FILE="$RUN_DIR/normalization_stats.json"
mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$SUMMARY_DIR"

echo "Run directory: $RUN_DIR"
"$PYTHON_BIN" "$HERE/vds2raw/audit.py" --data-root "$DATA_ROOT" \
  --stats-file "$STATS_FILE" --report-file "$RUN_DIR/dataset_audit.json"

run_one() {
  local tag="$1"; shift
  local log="$LOG_DIR/$tag.log"
  echo ">>> $tag"
  if "$@" > "$log" 2>&1; then
    echo "    completed; log: $log"
  else
    tail -n 40 "$log" || true
    echo "FAILED: $tag; log: $log"
    exit 1
  fi
}

for seed in "${SEEDS[@]}"; do
  COMMON=(--data-root "$DATA_ROOT" --save-dir "$ARTIFACT_DIR" \
    --stats-file "$STATS_FILE" --seed "$seed" --num-workers "${VDS2RAW_NUM_WORKERS:-2}" --skip-audit)
  run_one "ccsa_seed$seed" "$PYTHON_BIN" "$HERE/vds2raw/train_ccsa.py" \
    --epochs "$CCSA_EPOCHS" --batch-size "$CCSA_BATCH" "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}"
  run_one "dsne_seed$seed" "$PYTHON_BIN" "$HERE/vds2raw/train_dsne.py" \
    --epochs "$DSNE_EPOCHS" --batch-size "$DSNE_BATCH" --nlayers "$DSNE_LAYERS" \
    "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}"
  run_one "mcft3_seed$seed" "$PYTHON_BIN" "$HERE/vds2raw/train_mcft.py" \
    --epochs "$MCFT_EPOCHS" --batch-size "$MCFT_BATCH" --eval-frequency "$MCFT_EVAL" \
    "${COMMON[@]}" "${MAX_BATCH_ARGS[@]}"
done

"$PYTHON_BIN" "$HERE/vds2raw/summarize.py" \
  --results-dir "$ARTIFACT_DIR" --output-dir "$SUMMARY_DIR"
echo "All VDS2Raw methods completed. Summary: $SUMMARY_DIR"

