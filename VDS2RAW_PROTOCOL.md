# VDS2Raw 三模态实验协议（MCFT / TIR）

## 数据与选模

- 类别：Cargo、Fishing、Sailing & Pleasure。
- `source_train`：116；`target_train`：94；`target_val`：28；`target_test`：33。
- MCFT 按项目的全监督协议使用 `source_train + target_train`；TIR 按官方
  source-free few-shot 语义只使用 `target_train` 中每类 K 个带标签 support，禁止把
  source 数据拼入 TIR 训练。二者研究问题不同，不能再声称监督预算相同。
- MCFT 的归一化统计由两个训练分区拟合；TIR 的统计只由 `target_train` 拟合，避免
  source-free 适配阶段读取 source 数据。审计要求所有分区间 scene/MMSI 无交集。
- MCFT 沿用项目约定，每轮评估并按 `target_test` 峰值保存，因此存在明确的
  test-selection bias。TIR 官方默认只在 100 epoch 适配结束后评估和保存最终模型，
  不以中途 `target_test` 选模，也不开启早停。
- 每次运行保存 ACC、macro Precision/Recall/F1、最佳 epoch、最佳/最后检查点、
  完整逐轮训练与验证/测试曲线、逐样本预测和混淆矩阵。

## MCFT-RGB-NIR-AIS

MCFT 保留发布实现的双栅格 Transformer、ITA 与 ITM，并增加项目专用 AIS 后融合
分支。RGB/NIR 原生 `96×96` 输入先经过深度可分离卷积，再用
`AdaptiveAvgPool2d(15, 15)` 映射到发布模型要求的 `15×15` token 网格；不是先把
原图缩小到 `15×15`。

准备好的 NPZ 仍保留全部 13 个 AIS 字段用于审计，但模型加载边界会永久移除 9 个
navigation-status one-hot 字段。MCFT 只接收 `heading_sin`、`heading_cos`、`length_z`、
`width_z`；相应的 4 个 `ais_mask` 位只用于把缺失值门控为零，不拼入分类特征。这样
删除了 “engaged in fishing / under way sailing / using engine” 等几乎等价于类别的标签
代理。mask 是缺失信息，不计作第四模态。MCFT 原论文只包含两路栅格输入，AIS 分支
属于 VDS2Raw 适配，不宣称来自官方实现。

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
适配保留视觉 block 8/10 的 TIR、CoOp prompt 与 LoRA，并按数据集类别数改为
3-way 5-shot：每个 seed 从 `target_train` 每类确定性抽取 5 个 support，共 15 个样本；
不加载 `source_train` 训练样本或标签。多模态与数据加载部分是项目适配：

- RGB：原始 uint16 DN 先按 `target_train` 通道范围映射到 `[0,1]`，再使用官方 CLIP
  224×224 归一化与 ViT-B/16；禁止用 `/255` 后截断高位遥感信号；
- NIR：`target_train` 统计 z-score 后，通过零初始化门控残差注入 RGB；
- AIS：只使用与 MCFT 相同的 4 个基础字段，mask 仅作缺失值门控；
- 三分类仍逐字采用官方 `TIR/clip/model.py::TIR.forward` 的线性权重
  `1 - beta * (Sum - 3)`；官方源码明确把该分支标为 fewer classes/paper setting；
- CLIP 权重必须精确匹配全部键和形状，任何缺失、额外或 shape 不匹配都会报错，
  禁止静默漏载。

默认正式配置沿用官方核心超参数：100 epoch、完整 support set batch（配置值
`9999`，在 3-way 5-shot 下有效 batch 为 15，故每 epoch 恰好一次更新）、top-k ratio
0.3、temperature weight 0.5、视觉与文本双编码器 LoRA、rank 16 / alpha 8 /
dropout 0.25、LoRA AdamW lr `2e-4`、CoOp SGD lr `2e-3`、adapter SGD lr `1e-3`、
5-shot logit scale 4。官方 README 主命令不启用增强，因此默认 `--no-augment`；
`--augment` 只作为显式额外实验。正式默认 100 epoch 后一次评估、无早停。
VDS2Raw 的 NIR/AIS 分支及固定三分类测试划分不是论文原生设置，必须作为适配披露。
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

- `dataset_audit.json`、MCFT 的 `normalization_stats.json`、TIR 的
  `tir_target_normalization_stats.json`；
- `logs/<method>_seed<seed>.log`；
- `artifacts/*_best.pth`、`*_last.pth`、逐轮 CSV/JSON、预测与混淆矩阵；
- `summary/runs.csv`、`summary/summary.csv`（多 seed 时包含均值和总体标准差）。

历史 CCSA 和 d-SNE 已从当前实验入口移除。包含其他方法的混合历史汇总/总启动日志
不会作为这两个方法的专属日志删除，以免破坏 MCFT 等仍需保留的数据。
