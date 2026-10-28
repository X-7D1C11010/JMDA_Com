# VDS2Raw v2 三模态跨日有监督域适应

## 数据与协议

- 本地根目录：`D:\Downloads\VDS2Raw\threeclass_ready`
- 服务器根目录：`/home/lixiang/lx/VDS2Raw/threeclass_ready`
- 类别：Cargo、Fishing、Sailing & Pleasure
- 源域：`source_train`，day3，116 条
- 有标签目标训练域：`target_train`，非 day3，94 条
- 选模集：`target_val`，28 条
- 最终测试集：`target_test`，33 条

训练和数据集统计拟合严格限定为 `source_train + target_train`。每个方法只用
`target_val` 选择最佳 epoch；选定并重新载入最佳检查点后，才构造
`target_test` loader 并执行一次最终评价。最终保存 ACC、宏平均 Precision、
Recall 和 F1。

运行前的结构审计会检查 271 个 NPZ 的键、形状、类型、类别和 manifest
一致性，也会独立验证不同分区的 scene/MMSI 交集为零。该审计不计算训练统计、
不进行模型推理，也不影响选模。RGB/NIR 均值方差仅由 210 条训练样本计算；
AIS 中已经由数据构建程序基于相同训练分区标准化的 `length_z`、`width_z`
直接使用，并与 13 维缺失掩码一同送入模型。ImageNet 预训练的 CCSA/d-SNE
使用各自官方自然图像流程规定的输入归一化常量；这些外部预训练常量不从
`target_val` 或 `target_test` 拟合。

CCSA 与 d-SNE 对 `target_train` 使用同一规则：每个类别按随机种子固定抽取
10 条，共 30 条有标签目标样本。10-shot 是本 VDS2Raw 实验预先声明的数据
协议（d-SNE 的 VisDA-C 实验也使用 10-shot）；它不是对损失函数的修改。

## 方法输入

| 方法 | 输入 | 训练协议 |
|---|---|---|
| CCSA | RGB | ImageNet VGG-16 卷积层 + 1024/128 嵌入、CSA、双向分类更新 |
| d-SNE | RGB | ImageNet ResNetV2-101 + 4096/512 嵌入、官方跨域配对与 d-SNE 损失 |
| MCFT-RGB-NIR-AIS | RGB + NIR + AIS + mask | 官方双影像 MCFT 主干，增加显式 AIS/mask 后融合分支 |

MCFT 原论文只有两路栅格模态。因此三模态版本保留其 RGB/NIR Transformer、
ITA、ITM 和优化配方，同时用一个 26 维 `AIS*mask + mask` MLP 分支进行后
融合。这是透明标注的数据集适配版本，不宣称 AIS 分支来自原论文。

CCSA 先冻结 ImageNet VGG-16 卷积层，使用全部 `source_train` 初始化新加入的
全连接层 50 epoch，再解冻全部参数进行 80 epoch CCSA。每个 epoch 重新随机
生成 768 个受控配对（192 正、576 负），在三个目标类别间均分；有效 batch
size 为 256，显存中的 micro-batch 默认 16，梯度累积后才做一次优化更新。
这保留论文允许的随机配对选择和官方 1:3 正负比例，避免每个目标样本固定
遍历全部源样本。

d-SNE 先从 ImageNet ResNetV2-101 开始，在全部 `source_train` 上监督微调 50
epoch，再以 batch size 32、`--fn`、SGD 0.001、momentum 0.9、weight decay
1e-4 进行适配。10 epoch 是官方 Office-31 预算；程序继续到 20 epoch，并分别
以 `target_val` 在 1--10 和 1--20 范围选模，输出 `d-SNE@10` 与 `d-SNE@20`。
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
`VDS2RAW_MCFT_BATCH_SIZE` 覆盖。CCSA 还可以用
`VDS2RAW_CCSA_PAIRS_PER_EPOCH` 调整受控配对数（必须为 batch size 的整数倍），
用 `VDS2RAW_TARGET_SAMPLES_PER_CLASS` 同步调整两种方法的目标每类样本数。
适配训练轮数默认分别为 80/20/500。

首次正式运行会下载/读取 ImageNet 权重，并分别产生 CCSA 和 d-SNE 的源域
检查点。后续运行可复用这些检查点，跳过各自 50 epoch 的源域阶段：

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
export VDS2RAW_GPU_CANDIDATES=1
export VDS2RAW_METHOD_ORDER="dsne ccsa mcft3"
export VDS2RAW_GPU_STABLE_CHECKS=1
export VDS2RAW_GPU_POLL_SECONDS=2
bash scripts/run_vds2raw_background.sh formal
```

这会缩短方法切换空窗，但不能对不遵守本项目锁的其他用户提供 GPU 独占保证。训练代码
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

## 输出

每次运行写入 `runs/vds2raw/<运行ID>/`：

- `dataset_audit.json`：分区、NPZ、manifest 和泄漏审计；
- `normalization_stats.json`：只基于两个训练分区的 RGB/NIR 均值、标准差和原始 DN
  逐通道最小/最大值；MCFT 使用训练统计做 z-score。自然图像分支保留 RGB 色彩比例，
  固定除以 255 并裁剪稀疏高 DN 异常值（训练集各通道超过 255 的比例为 0.42%–1.46%）；
- `logs/`：各方法标准输出和错误日志；
- `artifacts/*_best.pth`：由 `target_val` 选择的最佳模型；
- `artifacts/*_last.pth`：最后一轮模型；
- `artifacts/*_source_init.pth`、`*_source_tuned.pth`：CCSA/d-SNE 源域初始化；
- `artifacts/*_target_val_metrics.csv/json`：验证曲线；
- `artifacts/*_result.csv/json`：选中验证指标和一次性最终测试指标；
- `summary/runs.csv`：每个方法、每个随机种子的最终四项测试指标；
- `summary/summary.csv`：跨随机种子的均值和总体标准差。
- `efficiency/`：当前 CCSA、d-SNE、三模态 MCFT 的参数量、MAC/GFLOPs、延迟、吞吐及
  训练 step/显存微基准；smoke 为快速验证，论文表应使用正式重复基准。

d-SNE 正式运行会同时生成 `d-SNE@10` 和 `d-SNE@20` 两行结果。

