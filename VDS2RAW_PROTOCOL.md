# VDS2Raw 三模态实验协议（MCFT / TIR）

## 数据与选模

- 类别：Cargo、Fishing、Sailing & Pleasure。
- `source_train`：116；`target_train`：94；`target_val`：28；`target_test`：33。
- 两种方法训练时均可见 `source_train + target_train` 的真实标签，监督预算一致。
- 归一化统计只由上述两个训练分区拟合；审计要求分区间 scene/MMSI 无交集。
- 按当前项目约定，每轮同时评估 `target_val` 与 `target_test`，最终结果取
  `target_test` ACC 最高、macro-F1 并列判优的 epoch。该做法存在明确的
  test-selection bias，不能作为无偏测试估计。
- 每次运行保存 ACC、macro Precision/Recall/F1、最佳 epoch、最佳/最后检查点、
  完整逐轮训练与验证/测试曲线、逐样本预测和混淆矩阵。

## MCFT-RGB-NIR-AIS

MCFT 保留发布实现的双栅格 Transformer、ITA 与 ITM，并增加项目专用 AIS 后融合
分支。RGB/NIR 原生 `96×96` 输入先经过深度可分离卷积，再用
`AdaptiveAvgPool2d(15, 15)` 映射到发布模型要求的 `15×15` token 网格；不是先把
原图缩小到 `15×15`。

AIS 与 JMDA 输入保持一致：13 个字段先由 `ais_mask` 门控，再与 13 个有效性位拼接
送入带 dropout 的后融合分支。mask 是 AIS 的缺失信息，不计作第四模态。MCFT 原论文
只包含两路栅格输入，AIS 分支属于 VDS2Raw 适配，不宣称来自官方实现。

默认正式配置为 batch size 32、200 epoch、成对轻量翻转、自然无放回打乱和早停；
类别/域均衡采样只保留为显式消融开关。损失、学习率及正则化参数均写入每次运行的
配置和结果元数据。

## TIR-RGB-NIR-AIS

官方来源：

- 仓库：<https://github.com/shuaiyi308/TIR>
- 固定提交：`188ff2fcb7373fd5882319212bf712a676e90773`
- 论文：Shuai Yi 等，*Addressing Exacerbated Attention Sink for Source-Free
  Cross-Domain Few-Shot Learning*，CVPR 2026。

`TIR/` 保存该提交的官方 CLIP、LoRA、prompt 与 TIR 主体；出处和授权状态见
`TIR/UPSTREAM.md`。固定提交未提供 LICENSE，因此不要在未取得作者许可时再分发。

官方实验是 CLIP ViT-B/16 上的 source-free、5-way 1/5-shot 目标域微调。VDS2Raw
适配保留视觉 block 8/10 的 TIR、CoOp prompt 与 LoRA，但按本项目要求改为三分类、
全监督、三模态训练：

- RGB：官方 CLIP 224×224 归一化与 ViT-B/16；
- NIR：训练集统计 z-score 后，通过零初始化门控残差注入 RGB；
- AIS：与 JMDA 一致，13 个值经有效性 mask 门控，并拼接 13 个 mask 位做后融合；
- 三分类 TIR 权重使用论文给出的通用简化规则；官方五分类线性规则仅在 K=5 时使用；
- CLIP 权重必须精确匹配全部键和形状，任何缺失、额外或 shape 不匹配都会报错，
  禁止静默漏载。

默认正式配置沿用官方核心超参数：100 epoch、top-k ratio 0.3、temperature weight
0.5、视觉与文本双编码器 LoRA、rank 16 / alpha 8 / dropout 0.25；VDS2Raw 的
NIR/AIS 适配器使用独立学习率。
所有官方协议差异均写入结果元数据。

若本地还没有官方 CLIP 权重，可在可联网机器执行以下命令。下载器同时校验官方
SHA-256 和完整结构，随后把 `pretrained_weights/clip/ViT-B-16.pt` 上传到服务器：

```bash
python tools/download_tir_clip.py
```

## 启动

服务器正式运行（默认依次运行 MCFT、TIR）：

```bash
cd /home/lixiang/lx/JMDA_Com
export VDS2RAW_DATA_ROOT=/home/lixiang/lx/VDS2Raw/threeclass_ready
export VDS2RAW_SEEDS="42 43 44 45 46"
export VDS2RAW_GPU_CANDIDATES="0,1"
export VDS2RAW_TIR_CLIP_CHECKPOINT=/home/lixiang/lx/JMDA_Com/pretrained_weights/clip/ViT-B-16.pt
bash scripts/run_vds2raw_background.sh formal
```

最小 smoke：

```bash
export VDS2RAW_SEEDS="42"
export VDS2RAW_RUN_COMPLEXITY_BENCHMARK=0
bash scripts/run_vds2raw_background.sh smoke
```

只运行一种方法可设置 `VDS2RAW_METHOD_ORDER="mcft3"` 或
`VDS2RAW_METHOD_ORDER="tir"`。GPU 门槛默认分别为 6144 MiB 与 14336 MiB，启动器会
等待候选卡满足空闲显存及利用率门槛；它不能替代集群级独占调度。

Windows 可运行：

```powershell
$env:PYTHON = 'D:\Anaconda\envs\pytorch\python.exe'
$env:VDS2RAW_DATA_ROOT = 'D:\Downloads\VDS2Raw\threeclass_ready'
./scripts/run_vds2raw_methods.ps1 -Mode smoke
```

## 输出

每次运行写入 `runs/vds2raw/<run-id>/`：

- `dataset_audit.json`、`normalization_stats.json`；
- `logs/<method>_seed<seed>.log`；
- `artifacts/*_best.pth`、`*_last.pth`、逐轮 CSV/JSON、预测与混淆矩阵；
- `summary/runs.csv`、`summary/summary.csv`（多 seed 时包含均值和总体标准差）。

历史 CCSA 和 d-SNE 已从当前实验入口移除。包含其他方法的混合历史汇总/总启动日志
不会作为这两个方法的专属日志删除，以免破坏 MCFT 等仍需保留的数据。
