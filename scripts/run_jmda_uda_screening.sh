#!/usr/bin/env bash
# Controlled JMDA diagnostic matrix. These runs are causal diagnostics only:
# none may be copied into the SDA/UDA comparison tables.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
JMDA_NET_DIR="${JMDA_NET_DIR:-$HERE/JMDA-Net}"
EPOCHS="${1:-${JMDA_SCREEN_EPOCHS:-30}}"
SEED="${JMDA_SEED:-42}"
NUM_ITERATIONS="${JMDA_NUM_ITERATIONS:-1}"
BATCH_SIZE="${JMDA_BATCH_SIZE:-32}"
NUM_WORKERS="${JMDA_NUM_WORKERS:-4}"
PROJECTION_DIM="${JMDA_PROJECTION_DIM:-64}"
SVD_BATCH_SIZE="${JMDA_SVD_STAT_BATCH_SIZE:-128}"
PREFETCH_FACTOR="${JMDA_PREFETCH_FACTOR:-2}"
PIN_MEMORY="${JMDA_PIN_MEMORY:-1}"
RUN_INVARIANTS="${JMDA_RUN_INVARIANT_TESTS:-1}"
EXPECTED_CODE_FINGERPRINT="8de7896a76fce67b"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="$HERE/server_logs/jmda_uda_screening/$RUN_ID"
LIST_ROOT="$HERE/prepared_lists/jmda_uda_screening_$RUN_ID"

if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || (( EPOCHS < 25 )); then
  echo "筛选实验至少需要25轮（10轮源域预热+10轮Tensor/SVD+5轮OT爬升）: $EPOCHS"
  exit 1
fi
if ! [[ "$NUM_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JMDA_NUM_ITERATIONS 必须是正整数: $NUM_ITERATIONS"
  exit 1
fi
if (( BATCH_SIZE != 32 || PROJECTION_DIM != 64 || SVD_BATCH_SIZE != 128 )); then
  echo "筛选矩阵锁定监督参考配置: batch=32 projection_dim=64 SVD_batch=128"
  echo "当前: batch=$BATCH_SIZE projection_dim=$PROJECTION_DIM SVD_batch=$SVD_BATCH_SIZE"
  exit 1
fi

ACTUAL_CODE_FINGERPRINT="$(
  "$PYTHON" "$HERE/tools/print_jmda_code_fingerprint.py" \
    --jmda-net-dir "$JMDA_NET_DIR"
)"
if [[ "$ACTUAL_CODE_FINGERPRINT" != "$EXPECTED_CODE_FINGERPRINT" ]]; then
  echo "JMDA 核心源码版本不匹配；尚未启动筛选实验。"
  echo "expected=$EXPECTED_CODE_FINGERPRINT actual=$ACTUAL_CODE_FINGERPRINT"
  exit 1
fi
echo "[OK] JMDA reviewed code fingerprint: $ACTUAL_CODE_FINGERPRINT"
if [[ "$PIN_MEMORY" = "1" ]]; then
  PIN_ARG="--pin_memory"
elif [[ "$PIN_MEMORY" = "0" ]]; then
  PIN_ARG="--no_pin_memory"
else
  echo "JMDA_PIN_MEMORY 必须为0或1: $PIN_MEMORY"
  exit 1
fi

read -r -a TARGETS <<< "${JMDA_TARGETS:-backlight}"
mkdir -p "$RUN_ROOT" "$LIST_ROOT"

echo "JMDA UDA diagnosis suite=$RUN_ID"
echo "targets=${TARGETS[*]} epochs=$EPOCHS iterations=$NUM_ITERATIONS seed=$SEED"
echo "output=$RUN_ROOT"
echo "注意：oracle pairing 使用目标训练样本真实类别建立对应，但目标CE始终为0；所有结果仅用于定位故障。"

"$PYTHON" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$LIST_ROOT" \
  --targets "${TARGETS[@]}"

"$PYTHON" "$HERE/tools/preflight_server.py" \
  --protocol jmda_oracle_shared_uda \
  --data-root "$DATA_ROOT" \
  --jmda-net-dir "$JMDA_NET_DIR" \
  --list-dir "$LIST_ROOT" \
  --targets "${TARGETS[@]}" \
  --check-all-paths \
  --require-cuda

if [[ "$RUN_INVARIANTS" = "1" ]]; then
  (cd "$HERE" && "$PYTHON" -m unittest \
    tools.test_jmda_protocol_wiring \
    tools.test_audit_jmda_uda_log \
    tools.test_summarize_jmda_screening)
  (cd "$JMDA_NET_DIR" && "$PYTHON" -m unittest test_tensor_ot_invariants.py)
fi

COMMON_ARGS=(
  --epochs "$EPOCHS"
  --num_iterations "$NUM_ITERATIONS"
  --base_seed "$SEED"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --projection_dim "$PROJECTION_DIM"
  --svd_projection_mode legacy_supervised
  --basis_transport_mode legacy_supervised
  --uda_source_warmup_epochs 10
  --svd_stat_batch_size "$SVD_BATCH_SIZE"
  --prefetch_factor "$PREFETCH_FACTOR"
  "$PIN_ARG"
  --no_auto_ablation_hparams
  --transport_mode sinkhorn
  --no_adaptive_sinkhorn
  --ot_sinkhorn_iterations 50
  --ot_warmup_epochs 10
  --ot_ramp_epochs 5
  --tensor_loss_weight 0.12
  --lr_feature 5e-6
  --lr_other 2e-4
  --weight_decay 5e-4
  --class_conditional_ot
  --balance_svd_classes
  --target_label_ratio 0
  --target_cls_weight 0
  --label_space_mode oracle_shared
  --visual_finetuning_mode legacy_supervised
  --no_sync_discriminator_lr
  --report_strategy last
  --diagnostic_screening
)

FAILURES=0

run_case() {
  local target="$1" target_cn="$2" support="$3" tag="$4"
  shift 4
  local log_path="$RUN_ROOT/${target}_${tag}.log"
  echo ">>> [$target/$tag] log=$log_path"
  if (
    cd "$JMDA_NET_DIR"
    echo "JMDA_SCREEN_EXPERIMENT=$tag"
    echo "JMDA_SCREEN_TARGET=$target"
    "$PYTHON" Ablation/module_ablation.py \
      --source_root "$DATA_ROOT/晴天" \
      --target_root "$DATA_ROOT/$target_cn" \
      --oracle_shared_labels "$support" \
      --target_train_pair_list "$LIST_ROOT/${target}_train_pairs.tsv" \
      "${COMMON_ARGS[@]}" \
      "$@"
  ) >"$log_path" 2>&1; then
    echo "    done: $target/$tag"
  else
    local exit_code=$?
    echo "    FAILED($exit_code): $target/$tag"
    tail -n 30 "$log_path" || true
    FAILURES=$((FAILURES + 1))
  fi
}

for target in "${TARGETS[@]}"; do
  case "$target" in
    backlight) target_cn="逆光" ;;
    rainy) target_cn="雨天" ;;
    foggy) target_cn="雾天" ;;
    dark) target_cn="黑天" ;;
    *) echo "未知目标域: $target"; exit 1 ;;
  esac
  support="$($PYTHON "$HERE/tools/read_target_class_support.py" \
    --manifest "$LIST_ROOT/paired_protocol_manifest.json" \
    --target "$target")"

  # A: no target forward, no SVD update, no Tensor alignment, no OT.
  run_case "$target" "$target_cn" "$support" \
    source_only_fixed_projection \
    --ablation_mode with_tensor_no_ot \
    --adv_loss_weight 0 \
    --diagnostic_source_only \
    --no_uda_pseudo_class_pairing

  # B/C: isolate pseudo-vs-oracle correspondence in Tensor/SVD; no adversarial/OT signal.
  run_case "$target" "$target_cn" "$support" \
    pseudo_tensor_only \
    --ablation_mode with_tensor_no_ot \
    --adv_loss_weight 0 \
    --uda_pseudo_class_pairing
  run_case "$target" "$target_cn" "$support" \
    oracle_tensor_only \
    --ablation_mode with_tensor_no_ot \
    --adv_loss_weight 0 \
    --diagnostic_oracle_pairing \
    --no_uda_pseudo_class_pairing

  # D/E: add the complete neural-OT path under pseudo vs true diagnostic correspondence.
  run_case "$target" "$target_cn" "$support" \
    pseudo_tensor_ot \
    --ablation_mode full \
    --adv_loss_weight 0.08 \
    --uda_pseudo_class_pairing
  run_case "$target" "$target_cn" "$support" \
    oracle_tensor_ot \
    --ablation_mode full \
    --adv_loss_weight 0.08 \
    --diagnostic_oracle_pairing \
    --no_uda_pseudo_class_pairing
done

"$PYTHON" "$HERE/tools/summarize_jmda_screening.py" \
  --log-dir "$RUN_ROOT" \
  --output-prefix "$RUN_ROOT/screening_summary"

echo "筛选实验结束：failures=$FAILURES"
echo "汇总: $RUN_ROOT/screening_summary.md"
echo "CSV:  $RUN_ROOT/screening_summary.csv"
exit "$FAILURES"
