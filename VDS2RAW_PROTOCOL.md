# VDS2Raw 三模态实验协议（MCFT / TIR）

## 数据与选模

- 类别：Cargo、Fishing、Sailing & Pleasure。
- `source_train`：116；`target_train`：94；`target_val`：28；`target_test`：33。
- 为保证对比监督预算一致，MCFT 与 TIR 都使用全部 `source_train + target_train`
  真实标签。TIR 因此是公平对比适配，不再宣称遵循论文的 source-free few-shot
  数据预算；其 TIR、CoOp、LoRA 核心和优化器超参数仍来自官方实现。
- 两种方法先读取 `target_train` 中实际有数据的类别，再用该集合过滤
  `source_train`；加载后强制检查 source/target 类别集合完全相同，否则立即报错。
- 两种方法的归一化统计均只由匹配类别后的训练分区拟合；审计要求所有分区间
  scene/MMSI 无交集。
- 两种方法每轮同时评估 `target_val` 与 `target_test`，以 `target_test` ACC 最高、
  macro-F1 并列判优的 epoch 作为本次 seed 的最终结果。该约定存在明确的
  test-selection bias，不能作为无偏测试估计。
- 每次运行保存 ACC、macro Precision/Recall/F1、最佳 epoch、完整逐轮训练与
  验证/测试曲线、逐样本预测和混淆矩阵。最佳模型状态只在内存中保留到最终评估，
  不写入 `.pth`；训练结束后不会残留模型文件。

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
配置和结果元数据。根据五 seed 欠拟合曲线，label smoothing 调为0.05、ITA权重调为
0.05、AIS dropout 调为0.3；主学习率仍为 `5e-4`。

## TIR-RGB-NIR-AIS

官方来源：

- 仓库：<https://github.com/shuaiyi308/TIR>
- 固定提交：`188ff2fcb7373fd5882319212bf712a676e90773`
- 论文：Shuai Yi 等，*Addressing Exacerbated Attention Sink for Source-Free
  Cross-Domain Few-Shot Learning*，CVPR 2026。

`TIR/` 保存该提交的官方 CLIP、LoRA、prompt 与 TIR 主体；出处和授权状态见
`TIR/UPSTREAM.md`。固定提交未提供 LICENSE，因此不要在未取得作者许可时再分发。

官方实验是 CLIP ViT-B/16 上的 source-free、5-way 1/5-shot 目标域微调。为满足本项目
的公平监督预算，VDS2Raw 适配使用匹配类别后的全部 source/target 训练样本（当前为
116+94=210），不再抽取每类5个 support。该数据预算是明确的项目偏离；视觉 block
8/10 的 TIR、CoOp prompt、LoRA 和官方学习率仍保留。多模态与数据加载部分是项目适配：

- RGB：文件虽使用 uint16 容器，但训练像素99%分位约为179/225/262，超过1000的值
  只是稀疏异常高值。因此按官方8-bit `ToTensor` 语义除以255并截断少量高值，再使用
  CLIP 224×224 归一化；避免按全局最大值缩放导致主体图像整体过暗；
- NIR：训练分区统计 z-score 后，通过零初始化门控残差注入 RGB；
- AIS：只使用与 MCFT 相同的 4 个基础字段，mask 仅作缺失值门控；
- 三分类仍逐字采用官方 `TIR/clip/model.py::TIR.forward` 的线性权重
  `1 - beta * (Sum - 3)`；官方源码明确把该分支标为 fewer classes/paper setting；
- CLIP 权重必须精确匹配全部键和形状，任何缺失、额外或 shape 不匹配都会报错，
  禁止静默漏载。

默认正式配置为100 epoch、batch size 32、每轮评估。针对当前210样本在约第30轮即
训练集100%的现象，保留 top-k ratio 0.3、temperature weight 0.5 及视觉/文本双编码器
LoRA，但将容量与更新幅度调为 rank 8 / alpha 4 / dropout 0.4、LoRA AdamW lr `1e-4`、
CoOp SGD lr `1e-3`、adapter SGD lr `5e-4`，加入 label smoothing 0.1、cosine衰减至
初始学习率的0.1，并默认开启RGB/NIR同步的轻裁剪与水平/垂直翻转。正式默认每轮评估、
取峰值、无早停。
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

- `dataset_audit.json`、共享的 `normalization_stats.json`；
- `logs/<method>_seed<seed>.log`；
- `artifacts/` 中的逐轮 CSV/JSON、预测与混淆矩阵；默认不生成或保留 `.pth`；
- `summary/runs.csv`、`summary/summary.csv`（多 seed 时包含均值和总体标准差）。

历史 CCSA 和 d-SNE 已从当前实验入口移除。包含其他方法的混合历史汇总/总启动日志
不会作为这两个方法的专属日志删除，以免破坏 MCFT 等仍需保留的数据。
