#!/usr/bin/env bash
# Foreground official single-modal suite with per-run idle-GPU scheduling.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-backlight}"
SEED="${2:-42}"
MODALITIES_RAW="${3:-vis,ir}"
METHODS_RAW="${4:-DANN,DeepCORAL,CDAN,MDD,MCC}"
PYTHON_BIN="${PYTHON:-/home/lixiang/envs/lx_pytorch/bin/python}"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT_DIR="${OFFICIAL_SINGLE_MODAL_OUT_DIR:-$HERE/efficiency_results/official_single_modal_$STAMP}"
LOG_DIR="$OUT_DIR/logs"
LIST_ROOT="$OUT_DIR/lists"
CHECKPOINT_DIR="$OUT_DIR/checkpoints"

export TORCH_HOME="${TORCH_HOME:-$HERE/pretrained_weights/torch}"
export JMDA_CLASS_NAMES_FILE="${JMDA_CLASS_NAMES_FILE:-$HERE/class.txt}"
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR" "$LIST_ROOT" "$CHECKPOINT_DIR"

IFS=',' read -r -a MODALITIES <<< "$MODALITIES_RAW"
IFS=',' read -r -a METHODS <<< "$METHODS_RAW"

case "$TARGET" in
  backlight|rainy|foggy|dark) ;;
  *) echo "Unsupported target: $TARGET" >&2; exit 2 ;;
esac

run_on_gpu() {
  local tag="$1"; shift
  local min_free="$1"; shift
  "$PYTHON_BIN" "$HERE/tools/run_on_available_gpu.py" \
    --min-free-mib "$min_free" \
    --max-util-percent "${JMDA_GPU_MAX_UTIL_PERCENT:-10}" \
    --stable-checks "${JMDA_GPU_STABLE_CHECKS:-3}" \
    --poll-seconds "${JMDA_GPU_POLL_SECONDS:-30}" \
    --gpus "${JMDA_GPU_CANDIDATES:-0,1}" \
    --tag "$tag" -- "$@"
}

common_args() {
  local modality="$1" list_dir="$2"
  printf '%s\0' \
    --target "$TARGET" --seed "$SEED" --modality "$modality" \
    --data-dir "$list_dir" --efficiency_output_dir "$OUT_DIR" \
    --save-dir "$CHECKPOINT_DIR" --num-workers "${UDA_NUM_WORKERS:-8}" \
    --eval-num-workers "${UDA_EVAL_NUM_WORKERS:-0}" \
    --prefetch-factor "${UDA_PREFETCH_FACTOR:-4}" \
    --worker-timeout-seconds "${UDA_WORKER_TIMEOUT_SECONDS:-300}" \
    --pin-memory --persistent-workers --no-save-model
}

for modality in "${MODALITIES[@]}"; do
  modality="${modality,,}"
  if [[ "$modality" != "vis" && "$modality" != "ir" ]]; then
    echo "Unsupported modality: $modality" >&2
    exit 2
  fi
  list_dir="$LIST_ROOT/$modality"
  mkdir -p "$list_dir"
  "$PYTHON_BIN" "$HERE/tools/prepare_paired_protocol_lists.py" \
    --data-root "$DATA_ROOT" --output-dir "$list_dir" \
    --targets "$TARGET" --modality "$modality"
  "$PYTHON_BIN" "$HERE/tools/preflight_server.py" \
    --protocol uda --data-root "$DATA_ROOT" \
    --jmda-net-dir "$HERE/JMDA-Net" --list-dir "$list_dir" \
    --targets "$TARGET" --required-modality "$modality" --check-all-paths

  for method in "${METHODS[@]}"; do
    mapfile -d '' -t common < <(common_args "$modality" "$list_dir")
    tag="${method}_${modality}_${TARGET}_seed${SEED}"
    log="$LOG_DIR/$tag.log"
    echo ">>> $tag (log=$log)"
    case "${method,,}" in
      dann)
        cmd=("$PYTHON_BIN" "$HERE/DANN/train_dann.py" "${common[@]}"
          --backbone resnet50 --epochs 20 --iters-per-epoch 1000
          --batch-size 32 --lr 0.01 --target-eval-interval 1)
        min_free=16000
        ;;
      deepcoral)
        backbone=resnet50
        extra=()
        if [[ "$modality" == "vis" ]]; then
          backbone=alexnet
          extra=(--alexnet-weights "$HERE/pretrained_weights/deepcoral_alexnet_owt_4df8aa71.pth")
        fi
        cmd=("$PYTHON_BIN" "$HERE/DeepCORAL/train_deepcoral.py" "${common[@]}"
          --backbone "$backbone" --epochs 20 --source-batch-size 200
          --target-batch-size 56 --test-batch-size 128 --lr 0.001
          --target-eval-interval 1 "${extra[@]}")
        min_free=18000
        ;;
      cdan)
        cmd=("$PYTHON_BIN" "$HERE/CDAN/train_cdan.py" "${common[@]}"
          --backbone ResNet50 --num-iterations 100004 --test-interval 500
          --batch-size 36 --lr 0.001 --method CDAN+E)
        min_free=18000
        ;;
      mdd)
        cmd=("$PYTHON_BIN" "$HERE/MDD/train_mdd.py" "${common[@]}"
          --backbone resnet50 --epochs 20 --iters-per-epoch 1000
          --batch-size 32 --lr 0.004 --trade-off 1
          --trade-off-warmup-epochs 0)
        min_free=20000
        ;;
      mcc)
        cmd=("$PYTHON_BIN" "$HERE/MCC/train_mcc.py" "${common[@]}"
          --backbone resnet50 --epochs 20 --iters-per-epoch 1000
          --batch-size 36 --lr 0.005 --temperature 2.5 --trade-off 1
          --source-warmup-epochs 0 --trade-off-warmup-epochs 0
          --label-smoothing 0)
        min_free=18000
        ;;
      *) echo "Unsupported method: $method" >&2; exit 2 ;;
    esac
    run_on_gpu "$tag" "$min_free" "${cmd[@]}" >"$log" 2>&1
    echo "    done: $tag"
  done
done

"$PYTHON_BIN" -c "import sys; sys.path.insert(0, r'$HERE/tools'); from efficiency_common import generate_summary_files; generate_summary_files(r'$OUT_DIR')"
echo "Official single-modal suite completed: $OUT_DIR"
