#!/usr/bin/env bash
# =============================================================================
# JMDA 服务器训练入口 —— SDA / UDA 两条协议严格分开，不混合；CoOp 单独 VLM。
#
# 用法:
#   bash scripts/run_server.sh sda [epochs]   # 监督域适应（论文主结果）
#   bash scripts/run_server.sh uda [epochs]   # 严格无监督域适应
#   bash scripts/run_server.sh vlm [epochs]   # CoOp（独立 VLM 表）
#
# 环境变量:
#   JMDA_DATA_ROOT  数据根目录（默认 /mnt/data/lixiang/data/Data）
#   PYTHON          Python 解释器（默认 python）
#
# 本仓库 JMDA-Net/ 子目录为本文方法（JMDA-Net）的源码镜像，与对比方法目录区分。
# 每次运行的日志 -> server_logs/<protocol>/，效率结果 -> efficiency_results/<protocol>/
# =============================================================================
set -uo pipefail

PROTOCOL="${1:-}"
EPOCHS="${2:-100}"
case "$PROTOCOL" in
  sda|uda|vlm) ;;
  *) echo "用法: bash scripts/run_server.sh {sda|uda|vlm} [epochs]"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
PYTHON="${PYTHON:-python}"
JMDA_NET_DIR="$HERE/JMDA-Net"

LOG_ROOT="$HERE/server_logs/$PROTOCOL"
EFF_DIR="$HERE/efficiency_results/$PROTOCOL"
export JMDA_DATA_ROOT="$DATA_ROOT"
export JMDA_EFFICIENCY_OUTPUT_DIR="$EFF_DIR"
mkdir -p "$LOG_ROOT" "$EFF_DIR"

TARGETS=(backlight rainy foggy dark)

if [ ! -d "$DATA_ROOT" ]; then
  echo "数据根不存在: $DATA_ROOT（可用 JMDA_DATA_ROOT 指定）"; exit 1
fi
echo "协议=$PROTOCOL 轮数=$EPOCHS 数据根=$DATA_ROOT"
echo "日志目录=$LOG_ROOT"

run() {  # run "<log-tag>" <cmd...>
  local tag="$1"; shift
  echo ">>> [$tag]"
  if "$@" > "$LOG_ROOT/$tag.log" 2>&1; then
    echo "    done: $tag"
  else
    echo "    FAILED: $tag (见 $LOG_ROOT/$tag.log)"
  fi
}

# 通用参数（各方法脚本一致）
COMMON=(--epochs "$EPOCHS" --efficiency_output_dir "$EFF_DIR")

if [ "$PROTOCOL" = "sda" ]; then
  # ---- SDA：所有方法使用同一源标签 + 同一目标标注子集（共享 list）----
  for t in "${TARGETS[@]}"; do
    run "source_only_$t"  "$PYTHON" "$HERE/SDA_Baselines/train_source_only_sda.py" --target "$t" "${COMMON[@]}"
    run "joint_$t"        "$PYTHON" "$HERE/SDA_Baselines/train_joint_training.py"  --target "$t" "${COMMON[@]}"
    run "target_only_$t"  "$PYTHON" "$HERE/SDA_Baselines/train_target_only.py"    --target "$t" "${COMMON[@]}"
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann_sda.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral_sda.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan_sda.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd_sda.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc_sda.py"               --target "$t" "${COMMON[@]}"
    run "modfinity_$t"    "$PYTHON" "$HERE/Modfinity/train_modfinity_sda.py"   --target "$t" "${COMMON[@]}"
    run "boomda_$t"       "$PYTHON" "$HERE/Boomda/train_boomda_sda.py"         --target "$t" "${COMMON[@]}"
    # JMDA-Net-SDA（本文方法，仓库内 JMDA-Net/ 目录）
    run "jmda_net_sda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && JMDA_SDA_LIST_DIR='$HERE/data_lists' '$PYTHON' main_sda.py --target '$t' --epochs '$EPOCHS'"
  done

elif [ "$PROTOCOL" = "uda" ]; then
  # ---- UDA：目标真值绝不参与训练/采样/损失/调参/早停；仅最终测试 ----
  for t in "${TARGETS[@]}"; do
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc.py"               --target "$t" "${COMMON[@]}"
    run "modfinity_$t"    "$PYTHON" "$HERE/Modfinity/train_modfinity.py"   --target "$t" "${COMMON[@]}"
    run "boomda_$t"       "$PYTHON" "$HERE/Boomda/train_boomda_jmda.py"    --target "$t" "${COMMON[@]}"
    run "transadapter_$t" "$PYTHON" "$HERE/TransAdapter/train_transadapter.py" --target "$t" "${COMMON[@]}"
    run "adclip_$t"       "$PYTHON" "$HERE/AD-CLIP/train_adclip.py"         --target "$t" --epochs "$EPOCHS"
    # JMDA-Net-UDA：目标标签比例=0、目标分类权重=0（本文方法，仓库内 JMDA-Net/ 目录）
    run "jmda_net_uda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && '$PYTHON' Ablation/module_ablation.py \
         --source_root '$DATA_ROOT/晴天' --target_root '$DATA_ROOT/$(
           case "$t" in backlight) echo 逆光;; rainy) echo 雨天;; foggy) echo 雾天;; dark) echo 黑天;; esac)' \
         --epochs '$EPOCHS' \
         --target_label_ratio 0 --target_cls_weight 0"
  done

else
  # ---- VLM 独立表：CoOp 不是 UDA 方法 ----
  for t in "${TARGETS[@]}"; do
    run "coop_$t" "$PYTHON" "$HERE/CoOp/train_coop.py" --target "$t" --epochs "$EPOCHS"
  done
fi

echo "全部任务结束（$PROTOCOL）。失败项见上方 FAILED 行。"
