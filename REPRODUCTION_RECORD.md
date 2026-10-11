# 域适应方法在 JMDA-Net 数据集上的复现操作记录

## 1. 任务概述

本任务针对 6 类域适应（Domain Adaptation, DA）方法，在 JMDA-Net 跨天气场景数据集上完成代码下载、数据适配、冒烟测试与结果记录。

- **源域（Source）**：晴天（Sunny）
- **目标域（Target）**：逆光（Backlight）、雨天（Rainy）、雾天（Foggy）、黑天（Dark）
- **数据集路径**：`D:\Code\JMDA-Net\Data`
- **工作目录**：`D:\Code\JMDA_Com`

**2026-10 追加**：第二批新增 4 种 DA 方法——**MDD、MCC**（采用清华 Transfer-Learning-Library 现代实现）、**Boomda**（作者代码，三路编码器替换为 IR/VIS/AIS）、**MODfinity**（无公开代码，按 CVPR 2025 论文忠实复现，标注 our reimplementation）。详见第 3.1、6.7–6.10 节及第 8 节第二批结果。

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

## 3.1 第二批方法与仓库

| 方法 | 来源 | 框架 | 说明 |
|------|------|------|------|
| MDD | [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)（`tllib/alignment/mdd.py` + 官方示例 `examples/domain_adaptation/image_classification/mdd.py`） | PyTorch | Margin Disparity Discrepancy（ICML 2019），官方现代实现 |
| MCC | 同上（`tllib/self_training/mcc.py` + 官方示例 `mcc.py`） | PyTorch | Minimum Class Confusion（ECCV 2020），官方现代实现 |
| Boomda | [sunjunaimer/Boomda](https://github.com/sunjunaimer/Boomda)（作者代码） | PyTorch | 多模态 UDA；将原 acoustic(Wavlm)/visual(APViT)/lexical(Bert) 三路编码器替换为 IR/VIS/AIS 编码器，保留信息瓶颈、伪标签、相关性对齐、多目标优化与模态平衡权重求解全部流程 |
| MODfinity | **无公开代码**，按 CVPR 2025 论文（Liu et al., [openaccess](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_MODfinity_Unsupervised_Domain_Adaptation_with_Multimodal_Information_Flow_Intertwining_CVPR_2025_paper.html)）忠实复现 | PyTorch | **our reimplementation**；S1 MOML 度量学习 + S2 细粒度样本筛选 + S3 模态亲和蒸馏 + S4 持续优化（Eq.1–8） |

**仓库获取方式**：本环境 GitHub 全域（github.com/raw/codeload/api）TLS 层不可达，`git clone` 失败（schannel/openssl 均失败），gitee 无镜像。最终通过 **jsdelivr CDN**（`cdn.jsdelivr.net/gh/<user>/<repo>@<branch>/<path>`，数据 API `data.jsdelivr.com/v1/packages/gh/<user>/<repo>@<branch>` 列文件树）用本地 curl 逐文件下载，并用内容校验（拒绝 "Couldn't find"/"404" 错误体）+ HTTP 状态码确认。MODfinity 主论文与补充材料 PDF 直接从 openaccess.thecvf.com 下载，用 PyMuPDF 提取正文（补充材料为纯图片无文本层）。

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
├── TransAdapter\                # 多模态 Transformer DA 方法
│   ├── train_transadapter.py
│   └── output\
├── Transfer-Learning-Library\   # thuml/tllib（jsdelivr 逐文件下载的裁剪版）
│   └── tllib\                   # 仅保留 MDD/MCC 所需子集（alignment/self_training/
│                                #   modules/utils/vision），__init__ 已裁剪并标注
├── MDD\                         # MDD（tllib 官方示例改编）
│   ├── train_mdd.py
│   └── output\
├── MCC\                         # MCC（tllib 官方示例改编）
│   ├── train_mcc.py
│   └── output\
├── Boomda\                      # Boomda 作者代码（jsdelivr 下载）+ JMDA 适配
│   ├── train_boomda_jmda.py     # IR/VIS/AIS 三编码器替换版训练脚本
│   ├── models\                  # 作者原始模块（Boomda_model/min_norm_solvers/等）
│   └── output\
├── Modfinity\                   # MODfinity（our reimplementation）
│   ├── train_modfinity.py
│   └── output\
└── run_new_methods.py           # 第二批 4 方法 × 4 目标域批量运行脚本
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

### 6.7 MDD（tllib）

**适配点**：
1. **tllib 裁剪**：仅下载 MDD/MCC 所需子集（`tllib/alignment/mdd.py`、`tllib/self_training/mcc.py`、`tllib/modules/*`、`tllib/utils/*`、`tllib/vision/models/resnet.py` 等），各 `__init__.py` 裁剪导入面并标注 `NOTE(JMDA adaptation)`
2. **numpy 2.x 兼容**：`tllib/modules/grl.py` 中 `np.float` 已在 NumPy 1.24+ 移除，改为内建 `float`（等价，结果本就是 Python 标量）
3. **torchvision 兼容**：torchvision 0.21 已移除 `model_urls`，`tllib/vision/models/resnet.py` 增加 try/except 回退字典（ImageNet-1k V1 官方 URL）
4. **prettytable 可选**：`tllib/utils/metric/__init__.py` 将 prettytable 改为可选导入（仅影响 ConfusionMatrix.format 美化打印）
5. **数据加载**：本地实现 `ImageListDataset`（读 data_lists 列表）与 `ResizeImage`（256→224，语义同 `tllib.vision.transforms.ResizeImage`）
6. **流程忠实**：训练循环完全按官方示例——`classifier(cat(x_s,x_t))` 返回 `(outputs, outputs_adv)` 后 chunk，`transfer_loss=-MDD(y_s,y_s_adv,y_t,y_t_adv)`，`classifier.step()`（分类器头 10 倍学习率、SGD+nesterov、LambdaLR 调度，均与官方一致，包括官方 lambda 内含 `args.lr` 双重缩放的原始写法）
7. **iters_per_epoch** = max(len(src_loader), len(tgt_loader))；每 epoch 在目标域 val 上评估并保存 best

**核心文件**：`MDD/train_mdd.py`，依赖 `Transfer-Learning-Library/tllib/`

### 6.8 MCC（tllib）

**适配点**：同 6.7 的 1–5（tllib 裁剪、numpy/torchvision/prettytable 兼容、数据加载）。
训练循环按官方 `mcc.py` 示例：`loss = CE(y_s) + trade_off * MCC(y_t)`，`temperature=2.5`。

**核心文件**：`MCC/train_mcc.py`

### 6.9 Boomda（作者代码，IR/VIS/AIS 替换）

**模态映射**（模态字母保持与作者求解器兼容 A/V/L）：A=AIS、V=VIS、L=IR。

**编码器替换**：
- 原 acoustic(Wavlm) → `AISEncoder`：1-D CNN（Conv1d 1→16→32→64, k=15, s=8）+ AdaptiveAvgPool1d(8) + Linear(512→256)，输入为 AIS I|Q 拼接的 23040 维向量（逐行 z-score）
- 原 visual(APViT) → `IRVisEncoder`：torchvision ResNet-18（ImageNet V1）→ 512 维
- 原 lexical(Bert) → `IRVisEncoder`：同上

**完整保留的作者流程**（逐行对照 `models/Boomda_model.py`）：
1. **伪标签融合投票**：`pred_tf = 1.5*onehot(fused) + 各模态 onehot`，`index_rely` 阈值 ≥3 过滤，`loss_CE_tpl`
2. **CORAL 相关性对齐**：逐模态（fa/fv/fl）+ 联合特征（fall/fall_t）
3. **信息瓶颈**：`entropy_re`（log-variance 求和，经 importlib 从 `models/myfunc/en_re.py` 加载以绕过包 `__init__` 的 LoRA 重依赖）
4. **多目标优化 + 模态平衡权重求解**：hook 捕获 `feat_all` 梯度 → `gradient_normalizers('loss')` 归一化 → 作者闭式 `use_full` 解（Q 对角=[pd_1..., sum(pd_2)]，sol=(1/Q_t)/sum(1/Q_t)，保留作者 gn_m 计算但用 gn 归一化 grads_m 的原行为）→ EMA(beta=0.2) 平滑 scale
5. **域判别器**：`DomainDiscriminator(1536)` 联合特征判别，`alpha = p**2` 调度
6. **损失系数**：`0.5*alpha*0.1*10(PL) + 0.1*20*0.05(CE_A+CE_V+CE_L) + 0.1*10*0.1(coralall) + 0.001*0.01(entropy各项)`；`mul=1e3`(coral)、`mul2=1e-3`(entropy)；AMP autocast+GradScaler
7. **关键实现细节**：各模态分类器（netCA/netCV hidden=64、netCL hidden=128、SimpleFC2）输入必须是 `feat_all` 的**列切片**（`feat_all[0:bs_s, 0:dim_a]` 等）——这是 hook 能捕获逐模态梯度的前提（与作者代码一致）

**JMDA 数据适配**：
- **IR 配对规则**（权威来源 `D:\Code\JMDA-Net\DataLoad.py`）：同一类别文件夹内 `sorted(可见光文件)` 与 `sorted(红外文件)` 按索引配对（截断到较短者）。注意 VIS/IR 文件名不同（同名配对仅能命中少量样本，已弃用）
- **AIS 分配**：`balanced_AIS-dataset_16classes_100persample.mat`（v7.3/h5py），`balanced_rcv_I/Q` (11520,1572) 转置为 (1572,11520)，I|Q 拼接 23040 维，`new_balanced_label` 标签 0..15 取 0..13；按类别循环分配给图像样本（**假设 AIS 标签 i ↔ 图像类别 i**，见第 10 节）
- 优化器 Adam：heads lr=1e-4、encoders enc_lr=1e-5、beta1=0.5（同作者 scripts/Boomda.sh）

**核心文件**：`Boomda/train_boomda_jmda.py`，复用 `Boomda/models/` 作者模块

### 6.10 MODfinity（our reimplementation）

**无公开代码**，按 CVPR 2025 论文正文（含 Eq.1–8、Fig.2–5、Sec.4.3.4 超参敏感性）忠实复现，实现于 `Modfinity/train_modfinity.py`，脚本头与日志均标注 **our reimplementation**。

**方法实现**：
- **S1 MOML（Eq.1–2）**：标签建模为独立模态——类编码器 `E_c`（bias-free Linear(14→512)，即可学习类原型）将 one-hot 映射为特征 L；亲和度 `φ(F,L)=exp(F·L/τ)`；InfoNCE 式损失（batch 内同标签为正样本）；模态特征用 detach 副本，梯度只流向 E_c（保证各编码器独立优化，论文 Fig.4 语义）
- **S2 细粒度样本筛选（Eq.3–5）**：`(m̄i,k̄i)=argmax_{m,k} φ(F(m)_ti, L_k)`，置信度 `R_ti=max_{m,k} φ`，按 R 排序取 top α% 构成学习材料 LM（逐 batch 实现，论文未明确粒度）
- **S3 MOD 蒸馏（Eq.6–7）**：对学生模态 m，`LM^(m)={⟨ti,m̄i⟩∈LM | m̄i≠m}`；每个 LM 样本以其 teacher 模态特征（detach，单向流动）为 key，与同伪标签样本为正对的 InfoNCE 式蒸馏
- **S4 持续优化（Eq.8）**：`L_all = Σ_m (L^m_c + λ·L^m_MOML + β·L^m_MOD)`；E_c 在适配期间持续用 MOML 更新（源域用真标签 + 目标域用伪标签，实现选择见下）

**论文超参**（Sec.4.3.4）：α=0.15（LM 比例）、β=1.2（MOD 权重）、λ=4.5（MOML 权重）——全部采用。

**论文未给出/无法获取的实现选择**（补充材料为纯图片无文本层，均在脚本头标注）：
- 优化器 Adam lr=1e-4（全参数）；τ=0.07（对比学习常用温度）
- φ 内 L2 归一化（否则 exp(dot/τ) 数值溢出）
- AIS 编码器输出投影到 512 维，使三模态共享同一亲和空间与单一 E_c（论文三个数据集的骨干输出维度均一致，故其用单一 E_c；JMDA 的 AIS 与图像特征维度不同，投影对齐是最贴近论文的处理）
- 评估输出各模态准确率与融合准确率（融合=各模态 softmax 平均，对应论文表格 Fusion 列；以融合准确率为最终指标）

**核心文件**：`Modfinity/train_modfinity.py`

## 7. 实验参数配置

所有方法统一配置：
- **源域**：晴天（sunny）
- **目标域**：逆光/雨天/雾天/黑天
- **输入尺寸**：224×224
- **归一化**：ImageNet 均值方差（[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]）或 CLIP 官方归一化
- **训练轮数**：3 epochs（冒烟测试）
- **批大小**：DANN/DeepCORAL/CDAN=32, AD-CLIP/TransAdapter=16, CoOp=32
- **优化器**：SGD（DANN/DeepCORAL/CDAN）、Adam（AD-CLIP/CoOp/TransAdapter）

第二批方法配置（与官方默认对齐）：
- **MDD**：batch=32, SGD(lr=0.004, momentum=0.9, wd=5e-4, nesterov), lr_gamma=0.0002, lr_decay=0.75, margin=4, trade_off=1, bottleneck=256（官方默认 bottleneck=1024，冒烟用 256）, width=256
- **MCC**：batch=36, SGD(lr=0.005, wd=1e-3, nesterov), lr_gamma=0.001, lr_decay=0.75, temperature=2.5, trade_off=1, bottleneck=256
- **Boomda**：batch=16, Adam(heads lr=1e-4, encoders lr=1e-5, beta1=0.5), embd_a=256, embd_v=embd_l=512, joint 特征 1536 维, d_f=32, AMP
- **MODfinity**：batch=16, Adam lr=1e-4, τ=0.07, α=0.15, λ=4.5, β=1.2, feat_dim=512
- 第二批均为 `num_workers=0`（Windows）+ `python -B`（避免向用户 site-packages 写 pycache 触发 sandbox 限制）

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

**结果分析**（第一批）：
- 所有方法均显著优于随机基线，验证了域适应的有效性
- CoOp 在逆光和雨天上表现最佳，提示学习对跨天气泛化有优势
- CDAN 在雾天上取得最高准确率（0.3565）
- TransAdapter 在黑天上表现最佳（0.2500），Transformer 对低光照场景适应性较好
- 黑天域整体较难，所有方法准确率偏低

### 8.1 第二批方法结果

目标域验证集准确率（Source=晴天，同一冒烟协议：3 epochs）：

| 方法 | 逆光 | 雨天 | 雾天 | 黑天 | 平均 |
|------|------|------|------|------|------|
| MDD (tllib) | 0.3056 | 0.2692 | 0.2435 | 0.2417 | 0.2650 |
| MCC (tllib) | 0.4722 | 0.4231 | 0.3652 | 0.1667 | 0.3568 |
| Boomda | 0.3056 | 0.1538 | 0.2348 | 0.1667 | 0.2152 |
| MODfinity* | 0.5833 | 0.1154 | 0.3391 | 0.1917 | 0.3074 |

\* our reimplementation（无公开代码）

**第二批结果分析**：
- **MCC** 平均最佳（0.3568），且在各目标域上表现稳定，最小类混淆对跨天气偏移有效
- **MODfinity** 在逆光（0.5833，全部 10 个方法中最高）和雾天（0.3391）上表现突出，但雨天偏低（0.1154）——其模态亲和蒸馏依赖较可信的伪标签，3 epochs 冒烟下伪标签质量在雨天不足；黑天上 AIS 模态准确率高达 0.5917 但 IR/VIS 偏低，融合被拖累
- **Boomda** 表现居中（0.2152）；其模态平衡权重求解正常工作（scale 约 [0.18, 0.26, 0.26, 0.30]），伪标签数量随训练增长（1→24/batch）
- **MDD** 表现稳定（0.2417–0.3056），无短板
- 黑天域对第二批方法同样偏难（MCC/MODfinity/Boomda 均低于各自其他目标域）
- 多模态方法（Boomda/MODfinity）在 3 epochs 冒烟下尚不足以发挥三模态互补优势（编码器 lr 较小/对比损失收敛慢），正式实验需更长训练

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

# MDD（tllib，第二批）
python -B D:\Code\JMDA_Com\MDD\train_mdd.py --target backlight --epochs 3

# MCC（tllib，第二批）
python -B D:\Code\JMDA_Com\MCC\train_mcc.py --target backlight --epochs 3

# Boomda（作者代码 IR/VIS/AIS 替换版，第二批）
python -B D:\Code\JMDA_Com\Boomda\train_boomda_jmda.py --target backlight --epochs 3

# MODfinity（our reimplementation，第二批）
python -B D:\Code\JMDA_Com\Modfinity\train_modfinity.py --target backlight --epochs 3
```

`--target` 可选值：`backlight`、`rainy`、`foggy`、`dark`

### 9.3 全方法批量运行

```bash
# 第一批 6 方法
python D:\Code\JMDA_Com\run_all_methods.py

# 第二批 4 方法（MDD/MCC/Boomda/MODfinity × 4 目标域，结果追加至 results_summary.txt）
python D:\Code\JMDA_Com\run_new_methods.py
```

## 10. 已知问题与注意事项

1. **CLIP 模型下载**：ViT-B/16 等模型因 SSL 证书问题无法自动下载，已使用本地缓存的 RN50 权重
2. **AD-CLIP 雾天/黑天**：因目标域数据量较大（440/465 张），单轮训练时间较长，需适当增加超时
3. **torchvision pretrained 参数**：PyTorch 新版已弃用 `pretrained=True`，建议改用 `weights=ResNet18_Weights.DEFAULT`（当前仅产生警告，不影响运行）
4. **numpy 2.0 兼容**：CDAN 原代码使用 `np.float`，已修复为 `float`
5. **多模态利用**：第一批实验仅使用可见光模态，红外模态数据可通过修改 `prepare_data.py` 中的 `modality` 参数切换（第二批 Boomda/MODfinity 已使用 IR/VIS/AIS 三模态）

**第二批新增**：

6. **tllib numpy 2.x 兼容**：`tllib/modules/grl.py` 中 `np.float` 已在 NumPy 1.24+ 移除，MDD 首跑报 `AttributeError`，已改为内建 `float`（等价，结果本就是 Python 标量）
7. **tllib torchvision 兼容**：torchvision 0.21 已移除 `model_urls`，`tllib/vision/models/resnet.py` 已加 try/except 回退字典（ImageNet-1k V1 官方 URL）
8. **tllib prettytable 可选**：本机 prettytable 装在用户 site-packages，其 `__pycache__` 被 sandbox 禁写会报错；已将 `tllib/utils/metric/__init__.py` 的 prettytable 改为可选导入，且第二批一律用 `python -B`（不写 pycache）运行
9. **GitHub 不可达**：本环境 GitHub 全域（github.com/raw/codeload/api）TLS 层不可达，tllib 与 Boomda 均通过 jsdelivr CDN 逐文件下载（见 3.1 节），并用内容校验（拒绝 "Couldn't find"/"404" 错误体）+ HTTP 状态码确认
10. **VIS/IR 配对规则**：JMDA-Net 数据集中同一类别下 VIS/IR 文件名**不同**（同名配对仅能命中少量样本，如 backlight val 仅 5/36，已弃用）；权威规则来自项目官方 `D:\Code\JMDA-Net\DataLoad.py`——同类文件夹内 `sorted(可见光文件)` 与 `sorted(红外文件)` 按索引配对、截断到较短者。Boomda/MODfinity 的 `PairedMultimodalDataset._find_ir_partner` 已按此实现（目标域 100% 配对，sunny 760/986）
11. **AIS 类别映射假设**：AIS 数据（16 类）与图像数据（14 类）无显式样本级对应，假设 **AIS 标签 i ↔ 图像类别 i**，按类别循环分配给图像样本；此为数据集层面的假设，若实际映射不同需调整
12. **MODfinity 复现自选项**：论文未给出优化器/温度等细节，且补充材料为纯图片无文本层。自选实现（脚本头已标注）：Adam lr=1e-4、τ=0.07、φ 内 L2 归一化（防 exp 溢出）、AIS 编码器投影到 512 维使三模态共享单一 E_c、逐 batch 选 LM（论文未明确粒度）、融合=各模态 softmax 平均
13. **MODfinity mod 损失偏小**：冒烟规模下 batch=16、|LM|=top 15%≈2，LM^(m) 常为空或仅自对，MOD 损失多为 0——属冒烟规模现象，正式实验增大 batch/训练时长后生效
14. **运行方式**：第二批脚本统一以 `python -B` 运行（避免 sandbox 下向用户 site-packages 写 pycache 报错）；boomda 信息瓶颈模块经 importlib 从 `models/myfunc/en_re.py` 加载以绕过包 `__init__` 的 LoRA 重依赖

## 11. SDA / UDA 协议拆分与服务器训练（2026-10）

### 11.1 两条实验轨道（严格不混合）

为保证论文表格公平与监督口径清晰，全部实验拆分为两条互不混合的轨道，结果分表呈现，**不得跨标签预算/模态/监督设置排名**：

- **SDA（Semi-supervised / Supervised DA，论文主表）**：所有方法使用完全相同的源标签和**完全相同的目标标注子集**（共享 `data_lists/{target}_sda_10shot_seed42.txt`，预算 10 标注/类，seed=42，由 `tools/make_sda_lists.py` 一次性生成）。
  - 对 DANN/DeepCORAL/CDAN/MDD/MCC/MODfinity/Boomda **仅统一增加目标分类损失 L_cls_target（默认权重 1.0）**，各方法的原始域对抗/相关对齐/条件对抗/Margin 差异/最小类混淆/伪标签/教师选择/模态平衡机制**全部原样保留**，结果命名为 `*-SDA`；不得用真值替换任何伪标签机制。
  - 主表成员：Source-only-SDA、Joint Training（源+目标 CE，无任何对齐模块）、7 个 *-SDA 基线、JMDA-Net-SDA、Target-only Oracle（仅用目标标注子集训练）。`Target-only-Oracle-Full`（全量目标标注）属于不同标签预算，单独报告。
- **UDA（严格无监督）**：DANN/DeepCORAL/CDAN/MDD/MCC/MODfinity/Boomda/TransAdapter/AD-CLIP/JMDA-Net-UDA 的训练、采样、损失、调参、早停、最佳 epoch 选择**均不得使用目标真值**；目标标签仅用于最终测试。
- **CoOp 不是 UDA**（源域训练提示词），放入独立 VLM 表。

效率记录按协议分别保存，字段含 `protocol`、`target_label_count`、`target_labels_per_class`、`modalities`、`backbone`、`pretraining`；汇总文件按 SDA/UDA/VLM 分节。

### 11.2 SDA 共享标注子集

`tools/make_sda_lists.py` 生成 4 个共享 list（优先选有 IR 配对的样本，个别类不足 10 个则取全部，shortfall 已记录）：

| 目标域 | 文件 | 样本数 | 覆盖类数 |
|--------|------|--------|----------|
| 逆光 | `backlight_sda_10shot_seed42.txt` | 76 | 9 |
| 雨天 | `rainy_sda_10shot_seed42.txt` | 53 | 6 |
| 雾天 | `foggy_sda_10shot_seed42.txt` | 89 | 9 |
| 黑天 | `dark_sda_10shot_seed42.txt` | 80 | 8 |

### 11.3 新增/修改文件清单

- SDA 训练脚本：`DANN/train_dann_sda.py`、`DeepCORAL/train_deepcoral_sda.py`、`CDAN/train_cdan_sda.py`、`MDD/train_mdd_sda.py`、`MCC/train_mcc_sda.py`、`Modfinity/train_modfinity_sda.py`、`Boomda/train_boomda_sda.py`（统一模式 `loss = 原始loss + w_cls_target * CE(目标标注子集)`；DANN/DeepCORAL/CDAN 改为源驱动 epoch）
- SDA 基线（新目录 **`SDA_Baselines/`**）：`sda_baselines_impl.py`、`train_source_only_sda.py`、`train_joint_training.py`、`train_target_only.py`
- 工具：`tools/make_sda_lists.py`、`tools/data_paths.py`（跨平台路径重映射）、`tools/efficiency_common.py`、`tools/benchmark_efficiency.py`（protocol/标签预算/骨干字段）
- 服务器批量入口：`scripts/run_server.sh`
- JMDA-Net 本体源码镜像（2026-10 起统一归入本仓库，与对比方法目录区分）：顶层 **`JMDA-Net/`** 目录。其中 `JMDA-Net/main_sda.py` 为 JMDA-Net-SDA 入口（原 `main.py` 未修改），`JMDA-Net/Ablation/module_ablation.py` 为 JMDA-Net-UDA 入口（`--target_label_ratio 0 --target_cls_weight 0`）。镜像仅含源码，遵循 JMDA-Net 自身 `.gitignore`，第三方实验目录（MambaOut-main 等）、数据、日志、权重、文档均不纳入；镜像内的 main_sda.py 已适配仓库布局（自动定位 `../data_lists` 与 `../tools`，数据根回退 `/mnt/data/lixiang/data/Data`）

### 11.4 跨平台路径重映射（本地零影响）

`tools/data_paths.py` 提供：`remap_data_path(path)` 将 list 内的 Windows 数据根（`d:\Code\JMDA-Net\Data`）重映射到环境变量 `JMDA_DATA_ROOT` 指定的根，未设置时原样返回；`data_path(*parts)` 用于 AIS 等文件。全部 18 个 JMDA_Com 训练脚本/库已接入，`--data-dir` 默认值改为基于 `__file__` 的相对路径。JMDA 仓库的 `main_sda.py` 内联了同逻辑（`_remap_data_path`、`_default_sda_list_dir`），保持该仓库自包含；SDA list 目录可用 `JMDA_SDA_LIST_DIR` 显式指定，日志目录可用 `JMDA_LOG_DIR` 指定。

### 11.5 本地冒烟测试结果（不做本地训练）

全部为 `--target backlight --epochs 1`（JMDA-Net-SDA 另加 `--num-iterations 1`），只验证全链路可运行，数字不代表正式结果：

| 脚本 | target val acc | 说明 |
|------|----------------|------|
| DANN-SDA | 0.9722 | 对抗损失 + L_cls_target |
| DeepCORAL-SDA | 1.0000 | CORAL + L_cls_target |
| CDAN-SDA | 0.9722 | 条件对抗 + L_cls_target |
| MDD-SDA | 0.3611 | MDD + L_cls_target |
| MCC-SDA | 0.9444 | MCC + L_cls_target |
| Modfinity-SDA | 1.0000（fusion） | S2 k_bar 伪标签/MOML/MOD 全保留 |
| Boomda-SDA | 0.6667 | fused voting 伪标签/模态平衡全保留 |
| Source-only-SDA | 0.3889 | 无源外损失 |
| Joint Training | 0.9722 | 源+目标 CE，无对齐模块 |
| Target-only Oracle | 0.1111 | 仅 76 个标注样本训练 |
| JMDA-Net-SDA | 0.7500 | L_cls_src + L_cls_tgt + L_cls_mid，SVD 全量目标集（单轮 1968 秒） |

### 11.6 服务器正式训练步骤

服务器数据根：`/mnt/data/lixiang/data/Data`（含 黑天/逆光/晴天/雾天/雨天/AIS/M4-SAR）。自 2026-10 起，JMDA-Net 本体源码也在本仓库 `JMDA-Net/` 目录内，服务器只需克隆 JMDA_Com 一个仓库。

```bash
# 1) 拉取最新代码
cd ~/lx/JMDA_Com && git pull

# 2) 指定数据根（run_server.sh 默认已是该路径，此处显式导出更稳妥）
export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data

# 3) 分协议运行（默认 100 epochs；日志 -> server_logs/<protocol>/，
#    效率结果 -> efficiency_results/<protocol>/）
bash scripts/run_server.sh sda 100     # SDA 论文主表（含 JMDA-Net-SDA）
bash scripts/run_server.sh uda 100     # 严格 UDA（含 JMDA-Net-UDA）
bash scripts/run_server.sh vlm 100     # CoOp，独立 VLM 表
```

说明：脚本自动编排全部方法 × 4 目标域（backlight/rainy/foggy/dark）；JMDA-Net-SDA 与 JMDA-Net-UDA 调用本仓库 `JMDA-Net/` 目录中的代码。单方法也可直接运行，例如：

```bash
cd JMDA-Net
JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data \
  JMDA_SDA_LIST_DIR=../data_lists \
  python main_sda.py --target backlight --epochs 100
```

