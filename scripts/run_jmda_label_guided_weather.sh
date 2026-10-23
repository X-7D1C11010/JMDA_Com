#!/usr/bin/env bash
# One-weather JMDA run: true target classes guide Tensor/SVD and OT, while
# target classification cross-entropy remains disabled.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
JMDA_NET_DIR="${JMDA_NET_DIR:-$HERE/JMDA-Net}"
TARGET="${JMDA_TARGET:-backlight}"
EPOCHS="${1:-${JMDA_EPOCHS:-30}}"
SEED="${JMDA_SEED:-42}"
NUM_ITERATIONS="${JMDA_NUM_ITERATIONS:-1}"
NUM_WORKERS="${JMDA_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${JMDA_PREFETCH_FACTOR:-2}"
PIN_MEMORY="${JMDA_PIN_MEMORY:-1}"
RUN_INVARIANTS="${JMDA_RUN_INVARIANT_TESTS:-1}"

# This first controlled run intentionally retains the reviewed supervised
# reference configuration. Change these only in a separately named ablation.
BATCH_SIZE=32
PROJECTION_DIM=64
SVD_BATCH_SIZE=128
EXPECTED_CODE_FINGERPRINT="83cf8634a962df2f"

if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || (( EPOCHS < 25 )); then
  echo "完整流程至少需要25轮（10轮源域预热+10轮Tensor/SVD+5轮OT爬升）: $EPOCHS"
  exit 1
fi
if ! [[ "$NUM_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JMDA_NUM_ITERATIONS 必须是正整数: $NUM_ITERATIONS"
  exit 1
fi

case "$TARGET" in
  backlight) TARGET_CN="逆光" ;;
  rainy) TARGET_CN="雨天" ;;
  foggy) TARGET_CN="雾天" ;;
  dark) TARGET_CN="黑天" ;;
  *) echo "JMDA_TARGET 必须是 backlight/rainy/foggy/dark: $TARGET"; exit 1 ;;
esac

case "$PIN_MEMORY" in
  1) PIN_ARG="--pin_memory" ;;
  0) PIN_ARG="--no_pin_memory" ;;
  *) echo "JMDA_PIN_MEMORY 必须为0或1: $PIN_MEMORY"; exit 1 ;;
esac

ACTUAL_CODE_FINGERPRINT="$(
  "$PYTHON" "$HERE/tools/print_jmda_code_fingerprint.py" \
    --jmda-net-dir "$JMDA_NET_DIR"
)"
if [[ "$ACTUAL_CODE_FINGERPRINT" != "$EXPECTED_CODE_FINGERPRINT" ]]; then
  echo "JMDA 核心源码版本不匹配；尚未启动训练。"
  echo "expected=$EXPECTED_CODE_FINGERPRINT actual=$ACTUAL_CODE_FINGERPRINT"
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="$HERE/server_logs/jmda_label_guided/$RUN_ID"
LIST_ROOT="$HERE/prepared_lists/jmda_label_guided_$RUN_ID"
CHECKPOINT_ROOT="$HERE/checkpoints/jmda_label_guided/$RUN_ID"
TRAIN_LOG="$RUN_ROOT/${TARGET}_label_guided_tensor_ot.log"
mkdir -p "$RUN_ROOT" "$LIST_ROOT" "$CHECKPOINT_ROOT"

echo "[OK] JMDA reviewed code fingerprint: $ACTUAL_CODE_FINGERPRINT"
echo "protocol=LABEL_GUIDED_DA_NO_TARGET_CE"
echo "target=$TARGET ($TARGET_CN) epochs=$EPOCHS iterations=$NUM_ITERATIONS seed=$SEED"
echo "output=$RUN_ROOT"
echo "training_log=$TRAIN_LOG"

"$PYTHON" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$LIST_ROOT" \
  --targets "$TARGET"

"$PYTHON" "$HERE/tools/preflight_server.py" \
  --protocol jmda_label_guided_da \
  --data-root "$DATA_ROOT" \
  --jmda-net-dir "$JMDA_NET_DIR" \
  --list-dir "$LIST_ROOT" \
  --targets "$TARGET" \
  --check-all-paths \
  --require-cuda

if [[ "$RUN_INVARIANTS" == "1" ]]; then
  (cd "$HERE" && "$PYTHON" -m unittest \
    tools.test_jmda_protocol_wiring \
    tools.test_audit_jmda_label_guided_log)
  (cd "$JMDA_NET_DIR" && "$PYTHON" -m unittest test_tensor_ot_invariants.py)
fi

SUPPORT="$(
  "$PYTHON" "$HERE/tools/read_target_class_support.py" \
    --manifest "$LIST_ROOT/paired_protocol_manifest.json" \
    --target "$TARGET"
)"

(
  cd "$JMDA_NET_DIR"
  "$PYTHON" Ablation/module_ablation.py \
    --ablation_mode full \
    --label_guided_alignment \
    --no_uda_pseudo_class_pairing \
    --source_root "$DATA_ROOT/晴天" \
    --target_root "$DATA_ROOT/$TARGET_CN" \
    --label_space_mode oracle_shared \
    --oracle_shared_labels "$SUPPORT" \
    --target_train_pair_list "$LIST_ROOT/${TARGET}_train_pairs.tsv" \
    --target_label_ratio 0 \
    --target_cls_weight 0 \
    --epochs "$EPOCHS" \
    --num_iterations "$NUM_ITERATIONS" \
    --base_seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    "$PIN_ARG" \
    --projection_dim "$PROJECTION_DIM" \
    --svd_stat_batch_size "$SVD_BATCH_SIZE" \
    --svd_projection_mode legacy_supervised \
    --basis_transport_mode legacy_supervised \
    --svd_max_sweeps 10 \
    --svd_tolerance 1e-3 \
    --uda_source_warmup_epochs 10 \
    --transport_mode sinkhorn \
    --no_adaptive_sinkhorn \
    --ot_sinkhorn_iterations 50 \
    --ot_warmup_epochs 10 \
    --ot_ramp_epochs 5 \
    --class_conditional_ot \
    --balance_svd_classes \
    --tensor_loss_weight 0.12 \
    --adv_loss_weight 0.08 \
    --lr_feature 5e-6 \
    --lr_other 2e-4 \
    --weight_decay 5e-4 \
    --visual_finetuning_mode legacy_supervised \
    --no_sync_discriminator_lr \
    --report_strategy last \
    --checkpoint_dir "$CHECKPOINT_ROOT" \
    --no_auto_ablation_hparams
) 2>&1 | tee "$TRAIN_LOG"

"$PYTHON" "$HERE/tools/audit_jmda_label_guided_log.py" \
  --log "$TRAIN_LOG" \
  --expected-iterations "$NUM_ITERATIONS"

echo "[OK] label-guided one-weather run completed"
echo "training_log=$TRAIN_LOG"
echo "checkpoints=$CHECKPOINT_ROOT"
