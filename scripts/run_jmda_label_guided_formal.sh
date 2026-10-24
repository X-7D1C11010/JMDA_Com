#!/usr/bin/env bash
# Formal JMDA experiment: four target weather domains x five seeds x 100 epochs.
# Target truth guides Tensor/SVD and class-conditional OT; target CE stays zero.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
JMDA_NET_DIR="${JMDA_NET_DIR:-$HERE/JMDA-Net}"
NUM_WORKERS="${JMDA_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${JMDA_PREFETCH_FACTOR:-2}"
PIN_MEMORY="${JMDA_PIN_MEMORY:-1}"
RUN_INVARIANTS="${JMDA_RUN_INVARIANT_TESTS:-1}"

EPOCHS=100
NUM_ITERATIONS=5
BASE_SEED=42
TARGETS=(backlight rainy foggy dark)
BATCH_SIZE=32
PROJECTION_DIM=64
SVD_BATCH_SIZE=128
EXPECTED_CODE_FINGERPRINT="8de7896a76fce67b"

case "$PIN_MEMORY" in
  1) PIN_ARG="--pin_memory" ;;
  0) PIN_ARG="--no_pin_memory" ;;
  *) echo "JMDA_PIN_MEMORY 必须为0或1: $PIN_MEMORY"; exit 1 ;;
esac
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "JMDA_NUM_WORKERS 必须为非负整数: $NUM_WORKERS"; exit 1
fi
if ! [[ "$PREFETCH_FACTOR" =~ ^[1-9][0-9]*$ ]]; then
  echo "JMDA_PREFETCH_FACTOR 必须为正整数: $PREFETCH_FACTOR"; exit 1
fi

"$PYTHON" -c "import thop; print('[OK] thop FLOPs profiler available')"
ACTUAL_CODE_FINGERPRINT="$(
  "$PYTHON" "$HERE/tools/print_jmda_code_fingerprint.py" \
    --jmda-net-dir "$JMDA_NET_DIR"
)"
if [[ "$ACTUAL_CODE_FINGERPRINT" != "$EXPECTED_CODE_FINGERPRINT" ]]; then
  echo "JMDA 核心源码版本不匹配；尚未启动正式训练。"
  echo "expected=$EXPECTED_CODE_FINGERPRINT actual=$ACTUAL_CODE_FINGERPRINT"
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="$HERE/server_logs/jmda_label_guided_formal/$RUN_ID"
LIST_ROOT="$HERE/prepared_lists/jmda_label_guided_formal_$RUN_ID"
CHECKPOINT_ROOT="$HERE/checkpoints/jmda_label_guided_formal/$RUN_ID"
EFFICIENCY_ROOT="$HERE/efficiency_results/jmda_label_guided_formal/$RUN_ID"
RESULT_ROOT="$HERE/results/jmda_label_guided_formal/$RUN_ID"
mkdir -p \
  "$RUN_ROOT" "$LIST_ROOT" "$CHECKPOINT_ROOT" \
  "$EFFICIENCY_ROOT" "$RESULT_ROOT"

echo "protocol=LABEL_GUIDED_DA_NO_TARGET_CE"
echo "targets=${TARGETS[*]} epochs=$EPOCHS iterations=$NUM_ITERATIONS"
echo "seeds=42,43,44,45,46 code_fingerprint=$ACTUAL_CODE_FINGERPRINT"
echo "logs=$RUN_ROOT"
echo "efficiency=$EFFICIENCY_ROOT"
echo "results=$RESULT_ROOT"

"$PYTHON" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$LIST_ROOT" \
  --targets "${TARGETS[@]}"

"$PYTHON" "$HERE/tools/preflight_server.py" \
  --protocol jmda_label_guided_da \
  --data-root "$DATA_ROOT" \
  --jmda-net-dir "$JMDA_NET_DIR" \
  --list-dir "$LIST_ROOT" \
  --targets "${TARGETS[@]}" \
  --check-all-paths \
  --require-cuda

if [[ "$RUN_INVARIANTS" == "1" ]]; then
  (cd "$HERE" && "$PYTHON" -m unittest \
    tools.test_jmda_protocol_wiring \
    tools.test_audit_jmda_label_guided_log \
    tools.test_summarize_jmda_label_guided_formal)
  (cd "$JMDA_NET_DIR" && "$PYTHON" -m unittest test_tensor_ot_invariants.py)
fi

for target in "${TARGETS[@]}"; do
  case "$target" in
    backlight) target_cn="逆光" ;;
    rainy) target_cn="雨天" ;;
    foggy) target_cn="雾天" ;;
    dark) target_cn="黑天" ;;
  esac
  support="$(
    "$PYTHON" "$HERE/tools/read_target_class_support.py" \
      --manifest "$LIST_ROOT/paired_protocol_manifest.json" \
      --target "$target"
  )"
  train_log="$RUN_ROOT/${target}_label_guided_tensor_ot.log"
  target_checkpoints="$CHECKPOINT_ROOT/$target"
  target_efficiency="$EFFICIENCY_ROOT/$target"
  mkdir -p "$target_checkpoints" "$target_efficiency"

  echo "[$target] start: target=$target_cn support=$support log=$train_log"
  (
    cd "$JMDA_NET_DIR"
    "$PYTHON" Ablation/module_ablation.py \
      --ablation_mode full \
      --label_guided_alignment \
      --no_uda_pseudo_class_pairing \
      --source_root "$DATA_ROOT/晴天" \
      --target_root "$DATA_ROOT/$target_cn" \
      --label_space_mode oracle_shared \
      --oracle_shared_labels "$support" \
      --target_train_pair_list "$LIST_ROOT/${target}_train_pairs.tsv" \
      --target_label_ratio 0 \
      --target_cls_weight 0 \
      --epochs "$EPOCHS" \
      --num_iterations "$NUM_ITERATIONS" \
      --base_seed "$BASE_SEED" \
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
      --checkpoint_dir "$target_checkpoints" \
      --efficiency_output_dir "$target_efficiency" \
      --efficiency_warmup_iterations 20 \
      --efficiency_benchmark_iterations 100 \
      --require_efficiency_flops \
      --no_auto_ablation_hparams
  ) 2>&1 | tee "$train_log"

  "$PYTHON" "$HERE/tools/audit_jmda_label_guided_log.py" \
    --log "$train_log" \
    --expected-iterations "$NUM_ITERATIONS" \
    --require-efficiency | tee -a "$train_log"
  echo "[$target] completed"
done

"$PYTHON" "$HERE/tools/summarize_jmda_label_guided_formal.py" \
  --log-dir "$RUN_ROOT" \
  --output-dir "$RESULT_ROOT" \
  --targets "${TARGETS[@]}" \
  --expected-iterations "$NUM_ITERATIONS"

echo "[OK] formal four-weather JMDA experiment completed"
echo "logs=$RUN_ROOT"
echo "checkpoints=$CHECKPOINT_ROOT"
echo "efficiency=$EFFICIENCY_ROOT"
echo "results=$RESULT_ROOT"
