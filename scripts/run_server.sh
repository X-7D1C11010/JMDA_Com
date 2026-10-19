#!/usr/bin/env bash
# =============================================================================
# JMDA 服务器训练入口 —— SDA / UDA / Supervised 三条协议严格分开，不混合。
#
# 用法:
#   bash scripts/run_server.sh sda [epochs]   # 监督域适应（论文主结果）
#   bash scripts/run_server.sh jmda_uda [epochs] # JMDA-Net 无监督
#   bash scripts/run_server.sh jmda_oracle_shared_uda [epochs] # 共同类 oracle 诊断（不进 UDA 主表）
#   bash scripts/run_server.sh uda [epochs]   # 其他无监督域适应方法
#   bash scripts/run_server.sh supervised [epochs] # MCFT（源域监督，VIS+IR，独立表）
#
# 环境变量:
#   JMDA_DATA_ROOT  数据根目录（默认 /mnt/data/lixiang/data/Data）
#   JMDA_NET_DIR    JMDA-Net 源码目录（默认使用本仓库 JMDA-Net/ 镜像）
#   PYTHON          Python 解释器（默认 python）
#   JMDA_SEED       当前运行的随机种子（默认 42；正式多种子实验由外层循环传入）
#   JMDA_NUM_ITERATIONS  每个目标域的连续随机种子数（默认 5）
#   JMDA_BATCH_SIZE  JMDA UDA 训练批次（默认 32）
#   JMDA_NUM_WORKERS JMDA UDA DataLoader worker 数（默认 0）
#   JMDA_PROJECTION_DIM  JMDA Tensor/SVD 每模态投影维度（默认 16；与已验证配置一致）
#   JMDA_UDA_SOURCE_WARMUP_EPOCHS  启动无标签对齐前的纯源域训练轮数（默认 10）
#   JMDA_SVD_STAT_BATCH_SIZE  JMDA UDA SVD 特征统计批次（默认 32）
#   JMDA_PREFETCH_FACTOR  每个 worker 的预取批次数（默认 2；workers=0 时不启用）
#   JMDA_PIN_MEMORY  是否为 JMDA UDA DataLoader 启用 pinned memory（0/1，默认 1）
#   JMDA_RUN_INVARIANT_TESTS  JMDA 训练前运行 Tensor/OT 数值门禁（0/1，默认 1）
#   JMDA_MIN_FREE_GPU_MEMORY_MIB  可选的启动前空闲显存门禁（MiB；默认不启用）
#   JMDA_CLASS_NAMES_FILE  14类真实英文名称文件（PDA 文本提示必需）
#   CLIP_CACHE_DIR  CLIP 权重缓存目录（PDA；默认 ~/.cache/clip）
#
# 本仓库 JMDA-Net/ 子目录为本文方法（JMDA-Net）的源码镜像，与对比方法目录区分。
# 每次运行的日志 -> server_logs/<protocol>/，效率结果 -> efficiency_results/<protocol>/
# =============================================================================
set -uo pipefail

PROTOCOL="${1:-}"
EPOCHS="${2:-100}"
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "epochs 必须是正整数: $EPOCHS"
  exit 1
fi
case "$PROTOCOL" in
  sda|jmda_uda|jmda_oracle_shared_uda|uda|supervised) ;;
  *) echo "用法: bash scripts/run_server.sh {sda|jmda_uda|jmda_oracle_shared_uda|uda|supervised} [epochs]"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
PYTHON="${PYTHON:-python}"
JMDA_NET_DIR="${JMDA_NET_DIR:-$HERE/JMDA-Net}"
SEED="${JMDA_SEED:-42}"
JMDA_NUM_ITERATIONS="${JMDA_NUM_ITERATIONS:-5}"
JMDA_BATCH_SIZE="${JMDA_BATCH_SIZE:-32}"
JMDA_NUM_WORKERS="${JMDA_NUM_WORKERS:-0}"
JMDA_PROJECTION_DIM="${JMDA_PROJECTION_DIM:-16}"
JMDA_UDA_SOURCE_WARMUP_EPOCHS="${JMDA_UDA_SOURCE_WARMUP_EPOCHS:-10}"
JMDA_SVD_STAT_BATCH_SIZE="${JMDA_SVD_STAT_BATCH_SIZE:-32}"
JMDA_PREFETCH_FACTOR="${JMDA_PREFETCH_FACTOR:-2}"
JMDA_PIN_MEMORY="${JMDA_PIN_MEMORY:-1}"
JMDA_RUN_INVARIANT_TESTS="${JMDA_RUN_INVARIANT_TESTS:-1}"
# Version lock: the lightweight gate recomputes the same portable fingerprint
# from all JMDA core files.  Update this constant only together with reviewed
# source changes; it is intentionally not overridable from the environment.
JMDA_EXPECTED_CODE_FINGERPRINT="7d95307f51fd4d5d"
JMDA_MIN_FREE_GPU_MEMORY_MIB="${JMDA_MIN_FREE_GPU_MEMORY_MIB:-}"

require_positive_integer() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name 必须是正整数: $value"
    exit 1
  fi
}

require_nonnegative_integer() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "$name 必须是非负整数: $value"
    exit 1
  fi
}

if [[ "$PROTOCOL" = "jmda_uda" || "$PROTOCOL" = "jmda_oracle_shared_uda" ]]; then
  require_positive_integer "JMDA_NUM_ITERATIONS" "$JMDA_NUM_ITERATIONS"
  if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
    echo "JMDA_SEED 必须是整数: $SEED"
    exit 1
  fi
  if ! [[ "$JMDA_EXPECTED_CODE_FINGERPRINT" =~ ^[0-9a-f]{16}$ ]]; then
    echo "JMDA_EXPECTED_CODE_FINGERPRINT 必须是16位小写十六进制: $JMDA_EXPECTED_CODE_FINGERPRINT"
    exit 1
  fi
  require_positive_integer "JMDA_BATCH_SIZE" "$JMDA_BATCH_SIZE"
  require_nonnegative_integer "JMDA_NUM_WORKERS" "$JMDA_NUM_WORKERS"
  require_positive_integer "JMDA_PROJECTION_DIM" "$JMDA_PROJECTION_DIM"
  require_nonnegative_integer \
    "JMDA_UDA_SOURCE_WARMUP_EPOCHS" "$JMDA_UDA_SOURCE_WARMUP_EPOCHS"
  if (( JMDA_UDA_SOURCE_WARMUP_EPOCHS >= EPOCHS )); then
    echo "JMDA_UDA_SOURCE_WARMUP_EPOCHS 必须小于总训练轮数 EPOCHS；否则该任务全程只有源域训练，不能标记为 UDA: warmup=$JMDA_UDA_SOURCE_WARMUP_EPOCHS epochs=$EPOCHS"
    exit 1
  fi
  require_positive_integer "JMDA_SVD_STAT_BATCH_SIZE" "$JMDA_SVD_STAT_BATCH_SIZE"
  require_positive_integer "JMDA_PREFETCH_FACTOR" "$JMDA_PREFETCH_FACTOR"
  case "$JMDA_PIN_MEMORY" in
    0) JMDA_PIN_MEMORY_ARG="--no_pin_memory" ;;
    1) JMDA_PIN_MEMORY_ARG="--pin_memory" ;;
    *)
      echo "JMDA_PIN_MEMORY 必须是 0 或 1: $JMDA_PIN_MEMORY"
      exit 1
      ;;
  esac
  case "$JMDA_RUN_INVARIANT_TESTS" in
    0|1) ;;
    *)
      echo "JMDA_RUN_INVARIANT_TESTS 必须是 0 或 1: $JMDA_RUN_INVARIANT_TESTS"
      exit 1
      ;;
  esac
  if ! JMDA_ACTUAL_CODE_FINGERPRINT="$(
    "$PYTHON" "$HERE/tools/print_jmda_code_fingerprint.py" \
      --jmda-net-dir "$JMDA_NET_DIR"
  )"; then
    echo "无法计算 JMDA 核心源码指纹；尚未启动训练。"
    exit 1
  fi
  if [ "$JMDA_ACTUAL_CODE_FINGERPRINT" != "$JMDA_EXPECTED_CODE_FINGERPRINT" ]; then
    echo "JMDA 核心源码版本不匹配；尚未启动训练。"
    echo "expected=$JMDA_EXPECTED_CODE_FINGERPRINT actual=$JMDA_ACTUAL_CODE_FINGERPRINT"
    echo "请完整同步 JMDA 核心文件及启动/审计工具，禁止混用旧服务器副本。"
    exit 1
  fi
  echo "[OK] JMDA reviewed code fingerprint: $JMDA_ACTUAL_CODE_FINGERPRINT"
fi
if [ -n "$JMDA_MIN_FREE_GPU_MEMORY_MIB" ]; then
  require_nonnegative_integer \
    "JMDA_MIN_FREE_GPU_MEMORY_MIB" "$JMDA_MIN_FREE_GPU_MEMORY_MIB"
fi

LOG_ROOT="$HERE/server_logs/$PROTOCOL"
EFF_DIR="$HERE/efficiency_results/$PROTOCOL"
RESULTS_DIR="$HERE/results"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
SUITE_ID="${JMDA_SUITE_ID:-$RUN_ID}"
CHECKPOINT_ROOT="$HERE/checkpoints/$PROTOCOL/$RUN_ID"
FAIR_LIST_DIR="$HERE/prepared_lists/$SUITE_ID"
CLASS_NAMES_FILE="${JMDA_CLASS_NAMES_FILE:-}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-}"
# SDA 目标标注预算（每类样本数），CCSA / d-SNE 与子集生成共用同一数值
SDA_LABELS_PER_CLASS="${JMDA_SDA_LABELS_PER_CLASS:-10}"
export JMDA_DATA_ROOT="$DATA_ROOT"
export JMDA_EFFICIENCY_OUTPUT_DIR="$EFF_DIR"
mkdir -p "$LOG_ROOT" "$EFF_DIR" "$RESULTS_DIR" "$CHECKPOINT_ROOT" "$FAIR_LIST_DIR"

read -r -a TARGETS <<< "${JMDA_TARGETS:-backlight rainy foggy dark}"
VALID_TARGETS=" backlight rainy foggy dark "
for target in "${TARGETS[@]}"; do
  if [[ "$VALID_TARGETS" != *" $target "* ]]; then
    echo "未知目标域: $target（允许: backlight rainy foggy dark）"
    exit 1
  fi
done

if [ ! -d "$DATA_ROOT" ]; then
  echo "数据根不存在: $DATA_ROOT（可用 JMDA_DATA_ROOT 指定）"; exit 1
fi
echo "协议=$PROTOCOL 轮数=$EPOCHS 数据根=$DATA_ROOT"
echo "日志目录=$LOG_ROOT"
echo "生成所有方法共享的精确 VIS/IR 配对样本清单..."
if ! "$PYTHON" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$FAIR_LIST_DIR" \
  --targets "${TARGETS[@]}"; then
  echo "共享配对样本清单生成失败；尚未启动训练。"
  exit 1
fi

PREFLIGHT_ARGS=(
  --protocol "$PROTOCOL"
  --data-root "$DATA_ROOT"
  --jmda-net-dir "$JMDA_NET_DIR"
  --list-dir "$FAIR_LIST_DIR"
  --targets "${TARGETS[@]}"
)
if [ "${JMDA_REQUIRE_CUDA:-1}" = "1" ]; then
  PREFLIGHT_ARGS+=(--require-cuda)
fi
if [ "${JMDA_CHECK_ALL_PATHS:-1}" = "1" ]; then
  PREFLIGHT_ARGS+=(--check-all-paths)
fi
if [ -n "$JMDA_MIN_FREE_GPU_MEMORY_MIB" ]; then
  PREFLIGHT_ARGS+=(
    --min-free-gpu-memory-mib "$JMDA_MIN_FREE_GPU_MEMORY_MIB"
  )
fi
echo "运行公共配置预检..."
if ! "$PYTHON" "$HERE/tools/preflight_server.py" "${PREFLIGHT_ARGS[@]}"; then
  echo "公共配置预检失败；尚未启动任何训练任务。"
  exit 1
fi

if [[ "$JMDA_RUN_INVARIANT_TESTS" = "1" ]] \
   && [[ "$PROTOCOL" = "jmda_uda" || "$PROTOCOL" = "jmda_oracle_shared_uda" ]]; then
  echo "运行 JMDA 协议接线门禁..."
  if ! (cd "$HERE" && "$PYTHON" -m unittest \
    tools.test_jmda_protocol_wiring \
    tools.test_audit_jmda_uda_log); then
    echo "JMDA 协议接线门禁失败；尚未启动训练。"
    exit 1
  fi
  echo "运行 JMDA Tensor/OT 数值门禁..."
  if ! (cd "$JMDA_NET_DIR" && "$PYTHON" -m unittest test_tensor_ot_invariants.py); then
    echo "JMDA Tensor/OT 数值门禁失败；尚未启动训练。"
    exit 1
  fi
fi

FAILURES=0

run() {  # run "<log-tag>" <cmd...>
  local tag="$1"; shift
  local log_path="$LOG_ROOT/${tag}_${RUN_ID}.log"
  local started finished elapsed exit_code status record_protocol
  local input_modalities model_selection label_space_relation
  local audit_protocol audit_exit audit_status
  echo ">>> [$tag]"
  started="$(date +%s)"
  exit_code=0
  status="success"
  if "$@" > "$log_path" 2>&1; then
    echo "    done: $tag"
  else
    exit_code=$?
    status="failed"
    echo "    FAILED: $tag (见 $log_path)"
    echo "    ---- error tail ----"
    tail -n 30 "$log_path" || true
    echo "    --------------------"
    FAILURES=$((FAILURES + 1))
  fi
  audit_protocol=""
  if [ "$status" = "success" ]; then
    case "$tag" in
      jmda_net_uda_*) audit_protocol="UDA" ;;
      jmda_net_oracle_shared_uda_*) audit_protocol="ORACLE_SHARED_UDA" ;;
    esac
  fi
  if [ -n "$audit_protocol" ]; then
    if "$PYTHON" "$HERE/tools/audit_jmda_uda_log.py" \
      --log "$log_path" \
      --expected-protocol "$audit_protocol" \
      --expected-projection-dim "$JMDA_PROJECTION_DIM" \
      --expected-warmup-epochs "$JMDA_UDA_SOURCE_WARMUP_EPOCHS" \
      --expected-total-epochs "$EPOCHS" \
      --expected-code-fingerprint "$JMDA_EXPECTED_CODE_FINGERPRINT" \
      --expected-num-iterations "$JMDA_NUM_ITERATIONS" \
      --expected-base-seed "$SEED" \
      >> "$log_path" 2>&1; then
      audit_status="$(grep -E 'JMDA log audit status:' "$log_path" | tail -n 1 || true)"
      echo "    protocol audit completed: $tag (${audit_status:-status unavailable})"
    else
      audit_exit=$?
      exit_code=$audit_exit
      status="failed"
      FAILURES=$((FAILURES + 1))
      echo "    FAILED protocol audit: $tag (见 $log_path)"
      tail -n 30 "$log_path" || true
    fi
  fi
  finished="$(date +%s)"
  elapsed=$((finished - started))
  record_protocol="$PROTOCOL"
  case "$record_protocol" in
    jmda_uda)
      record_protocol="uda"
      label_space_relation="partial-set"
      ;;
    jmda_oracle_shared_uda)
      record_protocol="oracle_shared_uda"
      label_space_relation="oracle-shared-support"
      ;;
    *)
      label_space_relation="partial-set"
      ;;
  esac
  case "$tag" in
    jmda_net_*|mcft_*) input_modalities="VIS+IR" ;;
    *) input_modalities="VIS" ;;
  esac
  case "$record_protocol" in
    uda|oracle_shared_uda|supervised) model_selection="final_checkpoint" ;;
    *) model_selection="method_declared" ;;
  esac
  if ! "$PYTHON" "$HERE/tools/collect_run_result.py" \
    --log "$log_path" \
    --suite-id "$SUITE_ID" \
    --results-dir "$RESULTS_DIR" \
    --protocol "$record_protocol" \
    --run-tag "$tag" \
    --status "$status" \
    --exit-code "$exit_code" \
    --duration-seconds "$elapsed" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --efficiency-dir "$EFF_DIR" \
    --input-modalities "$input_modalities" \
    --label-space-relation "$label_space_relation" \
    --model-selection "$model_selection" \
    >> "$log_path" 2>&1; then
    echo "    FAILED result collection: $tag (见 $log_path)"
    FAILURES=$((FAILURES + 1))
  fi
}

# 通用参数（各方法脚本一致）
COMMON=(--epochs "$EPOCHS" --seed "$SEED" --data-dir "$FAIR_LIST_DIR" --efficiency_output_dir "$EFF_DIR" --save-dir "$CHECKPOINT_ROOT")

if [ "$PROTOCOL" = "sda" ]; then
  # ---- SDA：所有方法使用同一源标签 + 同一目标标注子集（共享 list）----
  for t in "${TARGETS[@]}"; do
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann_sda.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral_sda.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan_sda.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd_sda.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc_sda.py"               --target "$t" "${COMMON[@]}"
    # CCSA / d-SNE 原作即为有监督域适应，使用同一目标标注子集
    run "ccsa_$t"         "$PYTHON" "$HERE/CCSA/train_ccsa.py"                 --target "$t" "${COMMON[@]}" --target-labels-per-class "$SDA_LABELS_PER_CLASS" --sda-seed "$SEED"
    run "dsne_$t"         "$PYTHON" "$HERE/d-SNE/train_dsne.py"                --target "$t" "${COMMON[@]}" --target-labels-per-class "$SDA_LABELS_PER_CLASS" --sda-seed "$SEED"
    # JMDA-Net-SDA（本文方法，仓库内 JMDA-Net/ 目录）
    run "jmda_net_sda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && JMDA_SDA_LIST_DIR='$FAIR_LIST_DIR' '$PYTHON' main_sda.py --target '$t' --epochs '$EPOCHS' --num-iterations '${JMDA_NUM_ITERATIONS:-5}' --sda-seed '$SEED'"
  done

elif [ "$PROTOCOL" = "jmda_uda" ]; then
  # ---- JMDA-Net UDA：单独放在 SDA 与其他 UDA 方法之间运行 ----
  echo "JMDA加载参数: batch=$JMDA_BATCH_SIZE workers=$JMDA_NUM_WORKERS projection_dim=$JMDA_PROJECTION_DIM source_warmup=$JMDA_UDA_SOURCE_WARMUP_EPOCHS SVD_batch=$JMDA_SVD_STAT_BATCH_SIZE prefetch=$JMDA_PREFETCH_FACTOR pin_memory=$JMDA_PIN_MEMORY"
  for t in "${TARGETS[@]}"; do
    case "$t" in
      backlight) target_cn="逆光" ;;
      rainy)     target_cn="雨天" ;;
      foggy)     target_cn="雾天" ;;
      dark)      target_cn="黑天" ;;
    esac
    run "jmda_net_uda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && '$PYTHON' Ablation/module_ablation.py \
         --source_root '$DATA_ROOT/晴天' --target_root '$DATA_ROOT/$target_cn' \
         --epochs '$EPOCHS' --num_iterations '$JMDA_NUM_ITERATIONS' \
         --batch_size '$JMDA_BATCH_SIZE' \
         --num_workers '$JMDA_NUM_WORKERS' \
         --projection_dim '$JMDA_PROJECTION_DIM' \
         --uda_source_warmup_epochs '$JMDA_UDA_SOURCE_WARMUP_EPOCHS' \
         --svd_stat_batch_size '$JMDA_SVD_STAT_BATCH_SIZE' \
         --prefetch_factor '$JMDA_PREFETCH_FACTOR' $JMDA_PIN_MEMORY_ARG \
         --ablation_mode full --no_auto_ablation_hparams \
         --transport_mode sinkhorn \
         --target_label_ratio 0 --target_cls_weight 0 \
         --label_space_mode partial_set --sync_discriminator_lr \
         --report_strategy last --base_seed '$SEED' \
         --checkpoint_dir '$CHECKPOINT_ROOT/jmda_net_uda_$t'"
  done

elif [ "$PROTOCOL" = "jmda_oracle_shared_uda" ]; then
  # ---- 共同类别支持 Oracle：读取目标训练类支持，仅作 partial-set 负迁移诊断 ----
  # 该协议使用独立日志/结果/checkpoint，严禁并入严格 UDA 主表。
  echo "JMDA加载参数: batch=$JMDA_BATCH_SIZE workers=$JMDA_NUM_WORKERS projection_dim=$JMDA_PROJECTION_DIM source_warmup=$JMDA_UDA_SOURCE_WARMUP_EPOCHS SVD_batch=$JMDA_SVD_STAT_BATCH_SIZE prefetch=$JMDA_PREFETCH_FACTOR pin_memory=$JMDA_PIN_MEMORY"
  for t in "${TARGETS[@]}"; do
    case "$t" in
      backlight) target_cn="逆光" ;;
      rainy)     target_cn="雨天" ;;
      foggy)     target_cn="雾天" ;;
      dark)      target_cn="黑天" ;;
    esac
    run "jmda_net_oracle_shared_uda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && '$PYTHON' Ablation/module_ablation.py \
         --source_root '$DATA_ROOT/晴天' --target_root '$DATA_ROOT/$target_cn' \
         --epochs '$EPOCHS' --num_iterations '$JMDA_NUM_ITERATIONS' \
         --batch_size '$JMDA_BATCH_SIZE' \
         --num_workers '$JMDA_NUM_WORKERS' \
         --projection_dim '$JMDA_PROJECTION_DIM' \
         --uda_source_warmup_epochs '$JMDA_UDA_SOURCE_WARMUP_EPOCHS' \
         --svd_stat_batch_size '$JMDA_SVD_STAT_BATCH_SIZE' \
         --prefetch_factor '$JMDA_PREFETCH_FACTOR' $JMDA_PIN_MEMORY_ARG \
         --ablation_mode full --no_auto_ablation_hparams \
         --transport_mode sinkhorn \
         --target_label_ratio 0 --target_cls_weight 0 \
         --label_space_mode oracle_shared --no_uda_source_reweighting \
         --sync_discriminator_lr \
         --report_strategy last --base_seed '$SEED' \
         --checkpoint_dir '$CHECKPOINT_ROOT/jmda_net_oracle_shared_uda_$t'"
  done

elif [ "$PROTOCOL" = "uda" ]; then
  # ---- 其他 UDA 对比方法 ----
  for t in "${TARGETS[@]}"; do
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc.py"               --target "$t" "${COMMON[@]}"
    # PDA：CLIP ViT-B/16 提示学习 + 伪标签（无目标标签）
    run "pda_$t"          "$PYTHON" "$HERE/PDA/train_pda.py"               --target "$t" "${COMMON[@]}" --class-names-file "$CLASS_NAMES_FILE" --clip-cache-dir "$CLIP_CACHE_DIR"
  done

else
  # ---- Supervised 独立表：MCFT 是源域监督多模态分类，不是域适应方法 ----
  for t in "${TARGETS[@]}"; do
    run "mcft_$t" "$PYTHON" "$HERE/MCFT/train_mcft.py" --target "$t" "${COMMON[@]}"
  done
fi

echo "阶段结束（$PROTOCOL）：失败任务数=$FAILURES。"
if [ "$FAILURES" -ne 0 ]; then
  exit 1
fi
echo "全部任务成功（$PROTOCOL）。"
