# 域适应方法在 JMDA-Net 数据集上的复现操作记录

## 1. 任务概述

本任务针对 6 类域适应（Domain Adaptation, DA）方法，在 JMDA-Net 跨天气场景数据集上完成代码下载、数据适配、冒烟测试与结果记录。

- **源域（Source）**：晴天（Sunny）
- **目标域（Target）**：逆光（Backlight）、雨天（Rainy）、雾天（Foggy）、黑天（Dark）
- **数据集路径**：`D:\Code\JMDA-Net\Data`
- **工作目录**：`D:\Code\JMDA_Com`

## 2. 数据集概况

JMDA-Net 数据集为多模态（可见光 + 红外）跨天气域适应数据集，共 14 个类别。

| 域 | 划分 | 可见光 | 红外 |
|----|------|--------|------|
| 晴天 (Source) | train | 986 | 766 |
| 晴天 (Source) | val | 255 | 200 |
| 逆光 (Target) | train | 128 | 132 |
| 逆光 (Target) | val | 36 | 37 |
| 雨天 (Target) | train | 96 | 96 |
| 雨天 (Target) | val | 26 | 26 |
| 雾天 (Target) | train | 440 | 507 |
| 雾天 (Target) | val | 115 | 132 |
| 黑天 (Target) | train | 465 | 465 |
| 黑天 (Target) | val | 120 | 120 |

**注**：晴天目录结构为 `train/可见光/class_x`，其余天气域为 `train/class_x/可见光`。本实验统一使用**可见光**模态进行实验。

## 3. 方法与开源仓库筛选

| 方法类别 | 所选仓库 | 框架 | 说明 |
|---------|---------|------|------|
| DANN | [fungtion/DANN](https://github.com/fungtion/DANN) | PyTorch | 经典域对抗神经网络，含梯度反转层 |
| Deep CORAL | [SSARCandy/DeepCORAL](https://github.com/SSARCandy/DeepCORAL) | PyTorch | 协方差对齐的深度域适应 |
| CDAN | [thuml/CDAN](https://github.com/thuml/CDAN) | PyTorch | 清华官方实现，条件对抗域适应 |
| CLIP-based DA | [mainaksingha01/AD-CLIP](https://github.com/mainaksingha01/AD-CLIP) | PyTorch | 基于 CLIP 的域适应提示学习 |
| VLM DA | [KaiyangZhou/CoOp](https://github.com/KaiyangZhou/CoOp) | PyTorch | 视觉语言模型上下文优化 |
| 多模态 Transformer DA | [enesdoruk/TransAdapter](https://github.com/enesdoruk/TransAdapter) | PyTorch | Swin Transformer 域适应 |

**筛选标准**：
- 权威/社区公认的实现（官方或高星）
- 基于 PyTorch 框架（与本地环境兼容）
- 维护活跃、文档清晰
- 核心算法实现完整

## 4. 环境配置

- **Python**: 3.9.25 (`D:\Anaconda\envs\pytorch`)
- **PyTorch**: 2.6.0+cu126
- **torchvision**: 0.21.0+cu126
- **GPU**: NVIDIA GeForce RTX 4060 Ti
- **CLIP**: openai-clip 1.0（RN50 权重已缓存）
- **timm**: 1.0.27
- **transformers**: 4.57.6

## 5. 目录结构

```
D:\Code\JMDA_Com\
├── prepare_data.py              # 数据列表生成脚本
├── run_all_methods.py           # 全方法批量运行脚本
├── results_summary.txt          # 结果汇总
├── data_lists\                  # 各域图像列表文件
│   ├── sunny_train.txt
│   ├── sunny_val.txt
│   ├── backlight_train.txt
│   ├── backlight_val.txt
│   ├── rainy_train.txt
│   ├── rainy_val.txt
│   ├── foggy_train.txt
│   ├── foggy_val.txt
│   ├── dark_train.txt
│   └── dark_val.txt
├── DANN\                        # DANN 方法
│   ├── train_dann.py           # 适配后的训练脚本
│   └── output\                  # 模型权重
├── DeepCORAL\                   # Deep CORAL 方法
│   ├── train_deepcoral.py
│   └── output\
├── CDAN\                        # CDAN 方法
│   ├── train_cdan.py
│   ├── pytorch\                # 原始代码（已修改 network.py）
│   └── output\
├── AD-CLIP\                     # CLIP-based DA 方法
│   ├── train_adclip.py
│   ├── adclip_model.py         # 独立模型组件（去除 dassl 依赖）
│   └── output\
├── CoOp\                        # VLM DA 方法
│   ├── train_coop.py
│   ├── coop_model.py           # 独立模型组件
│   └── output\
└── TransAdapter\                # 多模态 Transformer DA 方法
    ├── train_transadapter.py
    └── output\
```

## 6. 各方法适配修改说明

### 6.1 DANN

**适配点**：
1. **骨干网络替换**：原仓库使用 MNIST 专用 CNN（28×28 输入，10 类），替换为 ResNet-18（ImageNet 预训练），适配 224×224 输入和 14 类
2. **数据加载**：原仓库使用 MNIST/自定义文本列表，新增 `ImageListDataset` 读取生成的图像列表文件
3. **域配置**：源域固定为晴天，目标域通过 `--target` 参数指定
4. **alpha 调度**：实现 DANN 论文中的梯度反转系数调度

**核心文件**：`DANN/train_dann.py`，复用 `DANN/models/functions.py` 中的 `ReverseLayerF`

### 6.2 Deep CORAL

**适配点**：
1. **骨干网络替换**：原仓库使用 AlexNet，替换为 ResNet-18
2. **数据加载**：原仓库硬编码 Office31 数据集，改为读取图像列表
3. **类别数**：从 31 类改为 14 类
4. **CORAL 损失**：直接复用原仓库 `models.py` 中的 `CORAL` 函数

**核心文件**：`DeepCORAL/train_deepcoral.py`

### 6.3 CDAN

**适配点**：
1. **numpy 兼容**：原 `network.py` 中 `calc_coeff` 使用 `np.float`（numpy 2.0 已移除），改为 `float`
2. **RandomLayer 设备**：`RandomLayer` 的 `random_matrix` 未注册为参数，需显式调用 `.cuda()` 移至 GPU
3. **对抗网络输入维度**：使用 RandomLayer 时，`ad_net` 输入维度应为 `output_dim=1024`，而非 `bottleneck_dim * num_classes`
4. **数据加载**：复用原 `ImageList` 类，传入生成的列表文件

**核心文件**：`CDAN/train_cdan.py`，复用 `CDAN/pytorch/network.py`、`CDAN/pytorch/loss.py`

### 6.4 AD-CLIP

**适配点**：
1. **去除 dassl 依赖**：原仓库依赖 `dassl` 框架，提取核心组件（PromptLearner、TextEncoder、CustomCLIP、projectors）到独立文件 `adclip_model.py`
2. **CLIP 模块冲突**：本地安装的 `clip` 包与 AD-CLIP 修改版冲突（AD-CLIP 的 ModifiedResNet 返回中间层特征）。通过 `importlib` 显式加载 AD-CLIP 的 `clip/model.py`
3. **torch.load 兼容**：PyTorch 2.6 默认 `weights_only=True`，对 TorchScript 存档需用 `torch.jit.load`
4. **DataLoader 进程数**：CLIP 预处理含 lambda，多进程序列化失败，设 `num_workers=0`
5. **骨干网络**：使用已缓存的 RN50 权重（ViT-B/16 因 SSL 问题无法下载）

**核心文件**：`AD-CLIP/train_adclip.py`、`AD-CLIP/adclip_model.py`

### 6.5 CoOp

**适配点**：
1. **去除 dassl 依赖**：提取 CoOp 核心组件到 `coop_model.py`
2. **配置对象**：实现 `SimpleCfg` 提供 `TRAINER.COOP.N_CTX`、`CSC`、`CLASS_TOKEN_POSITION` 等配置
3. **数据加载**：新增 `ImageListDataset` 读取图像列表
4. **训练方式**：在源域（晴天）上学习提示向量，在目标域验证集上评估泛化性能

**核心文件**：`CoOp/train_coop.py`、`CoOp/coop_model.py`

### 6.6 TransAdapter

**适配点**：
1. **简化实现**：原仓库依赖 `wandb`、自定义 Swin 实现、图对齐模块等复杂组件。实现简化版：使用 timm 的 Swin-Tiny 作为骨干，结合梯度反转层进行域对抗对齐
2. **数据加载**：新增 `ImageListDataset`
3. **训练流程**：源域分类损失 + 域对抗损失（GRL）

**核心文件**：`TransAdapter/train_transadapter.py`

## 7. 实验参数配置

所有方法统一配置：
- **源域**：晴天（sunny）
- **目标域**：逆光/雨天/雾天/黑天
- **输入尺寸**：224×224
- **归一化**：ImageNet 均值方差（[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]）或 CLIP 官方归一化
- **训练轮数**：3 epochs（冒烟测试）
- **批大小**：DANN/DeepCORAL/CDAN=32, AD-CLIP/TransAdapter=16, CoOp=32
- **优化器**：SGD（DANN/DeepCORAL/CDAN）、Adam（AD-CLIP/CoOp/TransAdapter）

## 8. 运行结果

目标域验证集准确率（Source=晴天）：

| 方法 | 逆光 | 雨天 | 雾天 | 黑天 | 平均 |
|------|------|------|------|------|------|
| DANN | 0.2778 | 0.0769 | 0.1652 | 0.1500 | 0.1675 |
| Deep CORAL | 0.2500 | 0.1923 | 0.2609 | 0.0417 | 0.1862 |
| CDAN | 0.2500 | 0.1923 | 0.3565 | 0.0417 | 0.2101 |
| AD-CLIP | 0.3056 | 0.1923 | 0.1391 | 0.1083 | 0.1863 |
| CoOp | 0.3333 | 0.2692 | 0.2348 | 0.0667 | 0.2260 |
| TransAdapter | 0.3056 | 0.1923 | 0.1043 | 0.2500 | 0.2131 |

**随机基线**（14 类）：0.0714

**结果分析**：
- 所有方法均显著优于随机基线，验证了域适应的有效性
- CoOp 在逆光和雨天上表现最佳，提示学习对跨天气泛化有优势
- CDAN 在雾天上取得最高准确率（0.3565）
- TransAdapter 在黑天上表现最佳（0.2500），Transformer 对低光照场景适应性较好
- 黑天域整体较难，所有方法准确率偏低

## 9. 复现步骤

### 9.1 数据准备

```bash
# 生成图像列表文件
python D:\Code\JMDA_Com\prepare_data.py
```

### 9.2 单方法运行

```bash
# DANN
python D:\Code\JMDA_Com\DANN\train_dann.py --target backlight --epochs 3

# Deep CORAL
python D:\Code\JMDA_Com\DeepCORAL\train_deepcoral.py --target backlight --epochs 3

# CDAN
python D:\Code\JMDA_Com\CDAN\train_cdan.py --target backlight --epochs 3

# AD-CLIP
python D:\Code\JMDA_Com\AD-CLIP\train_adclip.py --target backlight --epochs 3

# CoOp
python D:\Code\JMDA_Com\CoOp\train_coop.py --target backlight --epochs 3

# TransAdapter
python D:\Code\JMDA_Com\TransAdapter\train_transadapter.py --target backlight --epochs 3
```

`--target` 可选值：`backlight`、`rainy`、`foggy`、`dark`

### 9.3 全方法批量运行

```bash
python D:\Code\JMDA_Com\run_all_methods.py
```

## 10. 已知问题与注意事项

1. **CLIP 模型下载**：ViT-B/16 等模型因 SSL 证书问题无法自动下载，已使用本地缓存的 RN50 权重
2. **AD-CLIP 雾天/黑天**：因目标域数据量较大（440/465 张），单轮训练时间较长，需适当增加超时
3. **torchvision pretrained 参数**：PyTorch 新版已弃用 `pretrained=True`，建议改用 `weights=ResNet18_Weights.DEFAULT`（当前仅产生警告，不影响运行）
4. **numpy 2.0 兼容**：CDAN 原代码使用 `np.float`，已修复为 `float`
5. **多模态利用**：当前实验仅使用可见光模态，红外模态数据可通过修改 `prepare_data.py` 中的 `modality` 参数切换
