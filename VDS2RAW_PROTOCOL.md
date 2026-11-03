# VDS2Raw v2 三模态跨日有监督域适应

## 数据与协议

- 本地根目录：`D:\Downloads\VDS2Raw\threeclass_ready`
- 服务器根目录：`/home/lixiang/lx/VDS2Raw/threeclass_ready`
- 类别：Cargo、Fishing、Sailing & Pleasure
- 源域：`source_train`，day3，116 条
- 有标签目标训练域：`target_train`，非 day3，94 条
- 诊断集：`target_val`，28 条
- 峰值选模集：`target_test`，33 条

训练和数据集统计拟合严格限定为 `source_train + target_train`。按用户指定的当前
VDS2Raw 实验协议，三个方法均在训练期间评价 `target_val` 和 `target_test`，并按
`target_test` 的 ACC、macro-F1 顺序选择峰值 epoch。该结果是测试集参与选模后的
峰值，存在明确的 test-selection bias，不能解释为无偏泛化估计。最终保存 ACC、
宏平均 Precision、Recall、F1、两套逐轮曲线、逐轮训练损失/LR、逐样本预测和
混淆矩阵。

运行前的结构审计会检查 271 个 NPZ 的键、形状、类型、类别和 manifest
一致性，也会独立验证不同分区的 scene/MMSI 交集为零。该审计不计算训练统计、
不进行模型推理，也不影响选模。RGB/NIR 均值方差仅由 210 条训练样本计算；
AIS 中已经由数据构建程序基于相同训练分区标准化的 `length_z`、`width_z`
直接使用。13 维缺失掩码只是数据有效性标记，用于将无效 AIS 字段置零，不作为
第四模态，也不再拼入分类器特征。ImageNet 预训练的 CCSA/d-SNE
使用各自官方自然图像流程规定的输入归一化常量；这些外部预训练常量不从
`target_val` 或 `target_test` 拟合。

CCSA、d-SNE 与 MCFT 使用相同的全监督预算：全部 116 条 `source_train` 和全部
94 条 `target_train` 的真实标签在训练中可见。CCSA/d-SNE 的
`--target-samples-per-class 0` 表示全部目标训练样本；正数参数仅保留给显式消融，
正式统一协议拒绝非零配置。MCFT 继续用两个训练分区的拼接数据。

## 方法输入

| 方法 | 输入 | 训练协议 |
|---|---|---|
| CCSA | RGB + NIR + AIS | ImageNet RGB VGG-16 + 轻量 NIR/AIS 残差适配，保留 1024/128 嵌入、CSA、双向分类更新 |
| d-SNE | RGB + NIR + AIS | ImageNet RGB ResNetV2-101 + 轻量 NIR/AIS 残差适配，保留 4096/512 嵌入与 d-SNE 损失 |
| MCFT-RGB-NIR-AIS | RGB + NIR + AIS | 官方双影像 MCFT 主干，增加正则化 AIS 后融合分支；mask 仅控制缺失值 |

MCFT 原论文只有 HSI/LiDAR 两路栅格模态，不包含 AIS 或 AIS mask。因此三模态
版本保留 RGB/NIR Transformer、ITA 和 ITM，同时增加 AIS 后融合分支。这是项目
适配，不宣称 AIS 分支来自原论文。默认 `continuous` AIS 配置只使用航向正余弦、
船长和船宽；`nav_engaged in fishing`、`nav_under way sailing` 等近似类别标签的
航行状态字段默认排除。mask 只做 `AIS*mask` 有效性门控，不显式送入 MLP。

当前 VDS2Raw MCFT 的小样本正则化默认包括：RGB/NIR 成对水平/垂直翻转（同时
按镜像方向修正 AIS heading sin/cos）、AIS 分支 0.6 模态 dropout、0.15 label
smoothing、Adam weight decay 5e-4，以及 source/target × 类别六组均衡采样。正式
预算为 200 epoch，每轮评价，最少训练 40 轮且 patience 20；旧 500 epoch 是对当前
小数据集明显过长的上限，已被覆盖。ITM hard-negative 掩码按标签是否相等计算，避免
0-based Cargo 标签被误当成 sentinel 排除；VDS2Raw 总损失为
`CE + 0.1*ITA + 0.1*ITM`，降低仅首轮较大的 ITM 对梯度的冲击。这些正则化属于
VDS2Raw 三模态适配，不反向修改 JMDA 的官方配方 VIS+IR 基线。

CCSA 先冻结 ImageNet VGG-16 卷积层，使用全部 `source_train` 初始化新加入的
全连接层以及 NIR/AIS 适配层 50 epoch，再解冻全部参数进行 80 epoch CCSA。RGB 仍走
官方自然图像 VGG-16 主干；NIR 经轻量卷积编码为 128 维残差，AIS 仅取连续字段并经
有效性 mask 门控后形成另一条 128 维残差。适配阶段完全使用发布代码
的固定穷举配对：保留所有同类正对，随机打乱不同类负对后至多截取正对数的 3 倍，
整个训练期间不重新生成 pair 表；有效 batch size 为 256，尾部不足一个 batch 的
pair 与发布代码的整数除法一样不参与更新。每个完整 batch 先执行 source→target，
再对同一 batch 执行 target→source 更新。micro-batch 仅用于等价梯度累积，不改变
有效 batch 或优化器更新次数。训练预处理为固定 resize/center-crop，不使用自定义
随机增强。

d-SNE 先从 ImageNet ResNetV2-101 开始，在全部 `source_train` 上监督微调 80
epoch：前 10 epoch 冻结 RGB backbone，训练 4096/512 分类头及 NIR/AIS 适配层，随后
联合微调；再以 batch size 32、`--fn`、task LR 2e-4、RGB backbone LR 2e-5、
momentum 0.9、weight decay 5e-4 对全部有标签 `target_train` 进行适配。NIR 经轻量
卷积编码为 512 维残差，连续
AIS 经门控后形成另一条 512 维残差；官方 RGB ResNetV2-101、4096/512 头、损失和
默认目标方向更新不变。10 epoch 是官方 Office-31 预算；程序继续到 20 epoch，并分别
以 `target_test` 在 1--10 和 1--20 范围选峰值，输出 `d-SNE@10` 与 `d-SNE@20`。
两者均在元数据中标记测试集选模偏差。
原仓库依赖 MXNet/CUDA 9；本项目是在当前 PyTorch/CUDA 环境中的等价训练端口，
数据读取和指标保存是 VDS2Raw 接口层修改。

## Windows 本地运行

推荐先运行完整审计和一批次冒烟测试：

```powershell
Set-Location D:\Code\JMDA_Com
$env:PYTHON = 'D:\Anaconda\envs\pytorch\python.exe'
$env:VDS2RAW_DATA_ROOT = 'D:\Downloads\VDS2Raw\threeclass_ready'
$env:VDS2RAW_SEEDS = '42'
./scripts/run_vds2raw_methods.ps1 -Mode smoke
```

正式运行：

```powershell
./scripts/run_vds2raw_methods.ps1 -Mode formal
```

如果希望关闭当前 PowerShell 窗口后继续训练，可使用独立隐藏进程：

```powershell
./scripts/run_vds2raw_background.ps1 -Mode formal
```

启动器会打印 PID、标准输出日志和错误日志路径。

正式脚本默认使用 CCSA/d-SNE/MCFT 有效批大小 256/32/32；可以用
`VDS2RAW_CCSA_BATCH_SIZE`、`VDS2RAW_DSNE_BATCH_SIZE` 和
`VDS2RAW_MCFT_BATCH_SIZE` 覆盖。CCSA 的 pair 数由官方穷举规则唯一确定，旧参数
`VDS2RAW_CCSA_PAIRS_PER_EPOCH` 不再影响训练。CCSA 和 d-SNE 的标签预算变量
`VDS2RAW_CCSA_TARGET_SAMPLES_PER_CLASS`、`VDS2RAW_DSNE_TARGET_SAMPLES_PER_CLASS`
均默认 0，即全部；正式启动会拒绝任一非零值。兼容变量
`VDS2RAW_TARGET_SAMPLES_PER_CLASS` 仅供显式消融。适配训练轮数默认分别为
80/20/200。三种方法均启用 `target_test` 峰值协议下的早停：CCSA 默认最少 20 轮、
patience 15；d-SNE 默认最少 15 轮、patience 8；MCFT 默认最少 40 轮、patience 20。
这些值分别可由对应的 `VDS2RAW_*_EARLY_STOPPING_PATIENCE` 和
`VDS2RAW_*_EARLY_STOPPING_MIN_EPOCHS` 环境变量覆盖，设 patience 为 0 可关闭。

d-SNE 的正式适配 batch 仍为官方 32；源域阶段单独使用 batch 16、80 epoch，先冻结
ImageNet backbone 对官方 4096/512/分类器头预热 10 epoch，再用 backbone/head 初始
学习率 0.001/0.01 联合微调，以余弦调度降低后期漂移，并恢复源域训练 ACC 最高
（CE 并列判据）的模型。源域采样和
适配 batch 默认类别均衡，适配 batch 保证三类均有
正配对支持；增强使用遥感轻增强。相关变量为
`VDS2RAW_DSNE_SOURCE_BATCH_SIZE`、`VDS2RAW_DSNE_SOURCE_HEAD_WARMUP_EPOCHS`、
`VDS2RAW_DSNE_SOURCE_HEAD_LR` 和 `VDS2RAW_DSNE_SOURCE_BACKBONE_LR`。
适配阶段仍不传 `--train-src`，保持官方 Office-31 d-SNE 默认方向。

2026-11-03 日志诊断后的 VDS2Raw 调优配置覆盖旧超参数：CCSA 的 source/adaptation
Adadelta 初始学习率分别为 0.5/0.1，并使用独立余弦衰减和类别加权 CE；source batch
32 下每轮约 4 次更新，源阶段累计有效步幅与旧 batch 256、LR 1.0 配置近似，而适配
batch 仍为 256，避免增加已经偏高的适配更新密度。d-SNE 适配学习率为 0.0002、
`alpha=0.2`、weight decay 0.0005、label smoothing 0.1，启用分类头 dropout 和余弦
衰减。CCSA/d-SNE 适配时预训练 RGB backbone 使用 task head/多模态适配层的 0.1 倍
学习率（分别为 0.01/0.1 与 0.00002/0.0002），余弦调度始终保持该比例，减少 source
表征被目标适配快速覆盖；MCFT 使用学习率 0.0002、weight decay 0.0005、StepLR(15, 0.7)、label
smoothing 0.15、AIS dropout 0.6 和 source/target × 类别的分组均衡采样。CCSA 的
source/target CE 也分别使用各域自身频数的类别权重。所有值均可用启动器中对应的
`VDS2RAW_*` 环境变量覆盖。

CCSA/d-SNE 的多模态适配层是为与本项目方法保持相同传感器输入而增加的数据集适配，
不是 CCSA 或 d-SNE 官方仓库的原生结构。mask 只门控 AIS 缺失字段，不作为第四模态。
由于新增了 NIR/AIS 参数，旧的 RGB-only 完整 source checkpoint 与当前模型结构不兼容；
严格加载会直接报错。ImageNet backbone 文件仍可严格加载，因为 RGB 主干结构未改变。

`VDS2RAW_METHOD_ORDER` 可以指定一个到三个不重复的方法。例如只做 MCFT 多随机种子
复现实验时设置为 `mcft3`；复杂度统计会自动限制为本次实际运行的方法。

首次正式运行会下载/读取 ImageNet 权重，并分别产生 CCSA 和 d-SNE 的源域
检查点。后续运行可复用这些检查点，跳过 CCSA 50 epoch 或 d-SNE 80 epoch 的源域阶段：

```powershell
$env:VDS2RAW_CCSA_SOURCE_CHECKPOINT = 'D:\path\to\vds2raw_ccsa_seed42_source_init.pth'
$env:VDS2RAW_DSNE_SOURCE_CHECKPOINT = 'D:\path\to\vds2raw_dsne_seed42_source_tuned.pth'
./scripts/run_vds2raw_methods.ps1 -Mode formal
```

检查点路径会写入 `run_config.json` 和方法结果元数据，便于追溯。不同随机种子
若需要严格独立的源域初始化，应分别运行或为每个种子提供对应检查点。

### 在本地下载 ImageNet 权重后上传服务器

服务器网络较慢时，可在本地一次性下载并导出两个可移植主干文件；该步骤不
需要读取 VDS2Raw 数据：

```powershell
Set-Location D:\Code\JMDA_Com
$env:PYTHON = 'D:\Anaconda\envs\pytorch\python.exe'
& $env:PYTHON tools\download_vds2raw_pretrained.py `
  --output-dir pretrained_weights
```

成功后会生成：

- `pretrained_weights/ccsa_vgg16_imagenet1k_v1_backbone.pth`；
- `pretrained_weights/dsne_resnetv2_101_a1h_in1k_backbone.pth`；
- `pretrained_weights/manifest.json`，包含文件大小、SHA-256 和依赖版本。

把整个 `pretrained_weights` 目录上传到服务器项目目录后，设置：

```bash
export VDS2RAW_CCSA_PRETRAINED_BACKBONE=/home/lixiang/lx/JMDA_Com/pretrained_weights/ccsa_vgg16_imagenet1k_v1_backbone.pth
export VDS2RAW_DSNE_PRETRAINED_BACKBONE=/home/lixiang/lx/JMDA_Com/pretrained_weights/dsne_resnetv2_101_a1h_in1k_backbone.pth
bash scripts/run_vds2raw_background.sh formal
```

需要在指定物理 GPU 上优先运行高显存方法时，可覆盖方法顺序和门控参数：

```bash
export VDS2RAW_GPU_CANDIDATES="0,1"
export VDS2RAW_METHOD_ORDER="dsne ccsa mcft3"
export VDS2RAW_GPU_STABLE_CHECKS=3
export VDS2RAW_GPU_POLL_SECONDS=30
export VDS2RAW_CCSA_MIN_FREE_MIB=7168
export VDS2RAW_DSNE_MIN_FREE_MIB=12288
export VDS2RAW_MCFT_MIN_FREE_MIB=4608
bash scripts/run_vds2raw_background.sh formal
```

调度器会在候选卡中选择满足当前方法门槛且空闲显存最多的卡。门槛来自当前多模态模型
正式 batch 单步峰值 reserved 加约 3 GiB：CCSA 3.8652 GiB→7168 MiB、d-SNE
8.8906 GiB→12288 MiB、MCFT 1.1348 GiB→4608 MiB。它不能对不遵守本项目锁的其他用户
提供 GPU 独占保证。训练代码
不会调用 `torch.cuda.empty_cache()`；单个方法内部从训练切换到验证时，模型、优化器及
PyTorch 缓存分配器仍保持 CUDA 上下文。真正的独占必须由集群调度器或管理员授权的
GPU exclusive-process 模式提供，不应使用无效张量把显存人为占满。

此时服务器不会访问网络下载 ImageNet 权重，仍会分别执行 50 epoch 的
`source_train` 初始化/微调。`VDS2RAW_*_PRETRAINED_BACKBONE` 是原始 ImageNet
主干；`VDS2RAW_*_SOURCE_CHECKPOINT` 是已经完成源域训练的完整模型，两类变量
不能同时为同一个方法设置。若本地已经完成源域训练，优先上传后者，可同时
省去服务器上的 50 epoch 源域阶段。

多随机种子示例：

```powershell
$env:VDS2RAW_SEEDS = '42 43 44 45 46'
./scripts/run_vds2raw_methods.ps1 -Mode formal
```

## Linux 服务器运行

```bash
python -m pip install -r requirements-vds2raw.txt
export VDS2RAW_DATA_ROOT=/home/lixiang/lx/VDS2Raw/threeclass_ready
export VDS2RAW_SEEDS="42"
bash scripts/run_vds2raw_background.sh smoke
bash scripts/run_vds2raw_background.sh formal
```

Linux 也可设置 `VDS2RAW_CCSA_SOURCE_CHECKPOINT` 和
`VDS2RAW_DSNE_SOURCE_CHECKPOINT` 复用源域检查点。

在扩展到 seed 42--46 前，先执行一次完整调优验证（默认仅 seed 42）：

```bash
cd /home/lixiang/lx/JMDA_Com
export VDS2RAW_DATA_ROOT=/home/lixiang/lx/VDS2Raw/threeclass_ready
export VDS2RAW_GPU_CANDIDATES="0,1"
bash scripts/run_vds2raw_tuned_pilot_background.sh
```

该入口固定按 `d-SNE -> CCSA -> MCFT` 顺序运行，重新训练 CCSA/d-SNE source 阶段，
严格使用本地 ImageNet backbone，关闭未改变的复杂度重复测试。可用
`VDS2RAW_PILOT_SEED` 改单个 seed；若设置了旧 source checkpoint 会直接拒绝启动，
避免新旧 source 调度混用。完成后必须检查 `summary/runs.csv` 中的 `early_peak`、
peak-to-last 回落和逐类混淆矩阵，再决定是否扩展五个 seed。

## 输出

每次运行写入 `runs/vds2raw/<运行ID>/`：

- `dataset_audit.json`：分区、NPZ、manifest 和泄漏审计；
- `normalization_stats.json`：只基于两个训练分区的 RGB/NIR 均值、标准差和原始 DN
  逐通道最小/最大值；MCFT 使用训练统计做 z-score。自然图像分支保留 RGB 色彩比例，
  固定除以 255 并裁剪稀疏高 DN 异常值（训练集各通道超过 255 的比例为 0.42%–1.46%）；
- `logs/`：各方法标准输出和错误日志；
- `artifacts/*_best.pth`：按用户指定的 `target_test` ACC、macro-F1 并列判据选择的峰值模型；
- `artifacts/*_last.pth`：最后一轮模型；
- `artifacts/*_source_init.pth`、`*_source_tuned.pth`：CCSA/d-SNE 源域初始化；
- `artifacts/*_target_val_metrics.csv/json`：验证曲线；
- `artifacts/*_result.csv/json`：选中 epoch 的验证/测试指标（明确标记
  `test_selection_bias=true`，不是无偏测试估计）；
- `summary/runs.csv`：每个方法、每个随机种子的最终四项测试指标；
- `summary/summary.csv`：跨随机种子的均值和总体标准差。
- 汇总表同时记录 planned/completed/selected epoch、峰值相对预算的位置、末轮指标及
  peak-to-last 回落。当预算至少 10 轮且最佳 epoch 位于前 25%（至少前 3 轮）时，
  汇总器输出 `[EARLY-PEAK]`，明确提示优化计划仍需复核；smoke 的单轮结果不会误报。
- `efficiency/`：当前 CCSA、d-SNE、三模态 MCFT 的参数量、MAC/GFLOPs、延迟、吞吐及
  训练 step/显存微基准；smoke 为快速验证，论文表应使用正式重复基准。

d-SNE 正式运行会同时生成 `d-SNE@10` 和 `d-SNE@20` 两行结果。

