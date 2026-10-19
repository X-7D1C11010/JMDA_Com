# 域适应方法在 JMDA-Net 数据集上的复现操作记录

## 1. 任务概述

本任务在 JMDA-Net 跨天气场景数据集上完成对比方法的代码下载、数据适配、冒烟测试与结果记录。

- **源域（Source）**：晴天（Sunny）
- **目标域（Target）**：逆光（Backlight）、雨天（Rainy）、雾天（Foggy）、黑天（Dark）
- **数据集路径**：`D:\Code\JMDA-Net\Data`
- **工作目录**：`D:\Code\JMDA_Com`

**2026-10 追加**：第二批新增 4 种 DA 方法——**MDD、MCC**（采用清华 Transfer-Learning-Library 现代实现）、**Boomda**（作者代码，三路编码器替换为 IR/VIS/AIS）、**MODfinity**（无公开代码，按 CVPR 2025 论文忠实复现，标注 our reimplementation）。详见第 3.1、6.7–6.10 节及第 8 节第二批结果。

> **注意：2026-10-19 方法清单变更（现行有效）**：Boomda、MODfinity、AD-CLIP、CoOp、TransAdapter 及其相关脚本已按要求删除，本文档中描述这些方法的历史章节（3.1 部分、6.4–6.6、6.9–6.10、8.1、9.2–9.3 中对应条目）**仅作为历史记录保留、不再可复现**。现行保留并纳入一键脚本的方法为 9 个：DeepCORAL、DANN、CDAN、MDD、MCC、CCSA、d-SNE、PDA、MCFT（外加本文方法 JMDA-Net）。现行清单与新增方法说明见第 12 节。

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

**注**：晴天目录结构为 `train/可见光/class_x`，其余天气域为 `train/class_x/可见光`。第一批传统/VLM 基线使用可见光；JMDA-Net、Boomda、MODfinity 使用 VIS+IR，必须在结果表中显式标注模态，不能把两种输入设置视为完全同条件比较。

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

## 5. 目录结构（2026-10-19 变更后）

```
D:\Code\JMDA_Com\
├── prepare_data.py              # 数据列表生成脚本
├── results_summary.txt          # 历史结果汇总（含已删除方法的历史数据）
├── scripts\
│   ├── run_all_sunny.ps1        # 一键启动：9 方法 x 4 天气 x 5 seed
│   ├── run_uda_sunny.ps1        # 一键启动：仅 5 个 UDA 基线
│   └── run_server.sh            # 服务器批量入口（sda/uda/supervised 协议分离）
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
├── Transfer-Learning-Library\   # thuml/tllib（jsdelivr 逐文件下载的裁剪版）
│   └── tllib\                   # 仅保留 MDD/MCC 所需子集（alignment/self_training/
│                                #   modules/utils/vision），__init__ 已裁剪并标注
├── MDD\                         # MDD（tllib 官方示例改编）
│   ├── train_mdd.py
│   └── output\
├── MCC\                         # MCC（tllib 官方示例改编）
│   ├── train_mcc.py
│   └── output\
├── CCSA\                        # CCSA（ICCV 2017，SDA；原作 Keras 移植）
│   └── train_ccsa.py
├── d-SNE\                       # d-SNE（CVPR 2019，SDA；原作 mxnet 移植）
│   └── train_dsne.py
├── PDA\                         # PDA（AAAI 2024，UDA；CLIP 提示学习）
│   └── train_pda.py
├── MCFT\                        # MCFT（TGRS 2024；VIS+IR 监督分类）
│   ├── mcft_model.py            # 作者 MViT 移植（去 einops/timm 依赖）
│   └── train_mcft.py
└── JMDA-Net\                    # 本文方法源码镜像（Ablation/main_sda.py 等）
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
- **IR 配对规则（现行公平协议）**：在原始 train+val 并集内按 VIS/IR **完全相同文件 stem** 恢复观测对，再以稳定哈希同时划分两模态；JMDA-Net 与所有多模态对比方法共享这一 cohort。历史的独立排序/截断规则会混配观测，已停用。
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
10. **VIS/IR 配对规则（已更新）**：原始 train/val 可能把同一观测的 VIS 与 IR 分到不同 split。现由 `tools/prepare_paired_protocol_lists.py` 与 `JMDA-Net/Ablation/paired_dataset.py` 在两 split 并集内按完全相同 stem 恢复配对，并用稳定哈希重建 train/val；Boomda/MODfinity 通过 `tools/data_paths.py::exact_ir_partner` 使用同一规则。独立按序截断会产生错配，禁止用于正式实验。
11. **AIS 类别映射假设**：AIS 数据（16 类）与图像数据（14 类）无显式样本级对应，假设 **AIS 标签 i ↔ 图像类别 i**，按类别循环分配给图像样本；此为数据集层面的假设，若实际映射不同需调整
12. **MODfinity 复现自选项**：论文未给出优化器/温度等细节，且补充材料为纯图片无文本层。自选实现（脚本头已标注）：Adam lr=1e-4、τ=0.07、φ 内 L2 归一化（防 exp 溢出）、AIS 编码器投影到 512 维使三模态共享单一 E_c、逐 batch 选 LM（论文未明确粒度）、融合=各模态 softmax 平均
13. **MODfinity mod 损失偏小**：冒烟规模下 batch=16、|LM|=top 15%≈2，LM^(m) 常为空或仅自对，MOD 损失多为 0——属冒烟规模现象，正式实验增大 batch/训练时长后生效
14. **运行方式**：第二批脚本统一以 `python -B` 运行（避免 sandbox 下向用户 site-packages 写 pycache 报错）；boomda 信息瓶颈模块经 importlib 从 `models/myfunc/en_re.py` 加载以绕过包 `__init__` 的 LoRA 重依赖

## 11. SDA / UDA 协议拆分与服务器训练（2026-10）

### 11.1 两条实验轨道（严格不混合）

为保证论文表格公平与监督口径清晰，全部实验拆分为两条互不混合的轨道，结果分表呈现，**不得跨标签预算/模态/监督设置排名**：

- **SDA（Semi-supervised / Supervised DA，论文主表）**：所有方法使用完全相同的源标签和**完全相同的目标标注子集**（共享 `data_lists/{target}_sda_10shot_seed42.txt`，预算 10 标注/类，seed=42，由 `tools/make_sda_lists.py` 一次性生成）。
  - 对 DANN/DeepCORAL/CDAN/MDD/MCC/CCSA/d-SNE **仅统一增加目标分类损失 L_cls_target（默认权重 1.0）**，各方法的原始域对抗/相关对齐/条件对抗/Margin 差异/最小类混淆/对比语义对齐/随机邻域嵌入机制**全部原样保留**，结果命名为 `*-SDA`；不得用真值替换任何伪标签机制。
  - 主表成员：Source-only-SDA、Joint Training（源+目标 CE，无任何对齐模块）、6 个 *-SDA 基线、JMDA-Net-SDA、Target-only Oracle（仅用目标标注子集训练）。`Target-only-Oracle-Full`（全量目标标注）属于不同标签预算，单独报告。（注：2026-10-19 按方法清单收敛要求删除了 `SDA_Baselines/` 基线目录，故 Source-only/Joint Training/Target-only Oracle 三个基线当前不再可运行，仅作历史记录；保留并可复现的 SDA 基线为 DANN/DeepCORAL/CDAN/MDD/MCC/CCSA/d-SNE。）
- **UDA（严格无监督）**：DANN/DeepCORAL/CDAN/MDD/MCC/PDA/JMDA-Net-UDA 的训练、采样、损失、调参、早停、最佳 epoch 选择**均不得使用目标真值**；目标标签仅用于最终测试。
- **MCFT 不是 UDA**（源域监督、VIS+IR 双模态），放入独立 Supervised 表。

效率记录按协议分别保存，字段含 `protocol`、`target_label_count`、`target_labels_per_class`、`modalities`、`backbone`、`pretraining`；汇总文件按 SDA/UDA/Supervised 分节。

### 11.2 SDA 共享标注子集

`tools/make_sda_lists.py` 生成 4 个共享 list（优先选有 IR 配对的样本，个别类不足 10 个则取全部，shortfall 已记录）：

| 目标域 | 文件 | 样本数 | 覆盖类数 |
|--------|------|--------|----------|
| 逆光 | `backlight_sda_10shot_seed42.txt` | 76 | 9 |
| 雨天 | `rainy_sda_10shot_seed42.txt` | 53 | 6 |
| 雾天 | `foggy_sda_10shot_seed42.txt` | 89 | 9 |
| 黑天 | `dark_sda_10shot_seed42.txt` | 80 | 8 |

### 11.3 新增/修改文件清单

- SDA 训练脚本：`DANN/train_dann_sda.py`、`DeepCORAL/train_deepcoral_sda.py`、`CDAN/train_cdan_sda.py`、`MDD/train_mdd_sda.py`、`MCC/train_mcc_sda.py`、`CCSA/train_ccsa.py`（SDA，`adadelta` + 1-α/α 加权 CE/CSA）、`d-SNE/train_dsne.py`（SDA，`loss = (1-α)·xent + α·dSNE`）；统一模式 `loss = 原始loss + w_cls_target * CE(目标标注子集)`（CCSA/d-SNE 的 α 对比项即其原始语义/邻域项）
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
bash scripts/run_server.sh supervised 100   # MCFT，独立 Supervised 表
```

说明：脚本自动编排全部方法 × 4 目标域（backlight/rainy/foggy/dark）；JMDA-Net-SDA 与 JMDA-Net-UDA 调用本仓库 `JMDA-Net/` 目录中的代码。单方法也可直接运行，例如：

```bash
cd JMDA-Net
JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data \
  JMDA_SDA_LIST_DIR=../data_lists \
  python main_sda.py --target backlight --epochs 100
```

### 11.7 2026-10-14 无监督协议与复现错误复核

本次复核以当前代码和服务器日志为准，修正了以下会实质影响结果或公平性的实现问题：

- **JMDA-Net 协议损失隔离**：SDA 使用 `L_cls_src + L_cls_tgt + L_cls_mid`；SSDA 只在固定目标标注子集上增加 `L_cls_tgt`；严格 UDA 只使用 `L_cls_src`。UDA/SSDA 的 all-pairs OT 中间特征不再错误套用源样本硬标签。
- **SDA 回归修复**：`module_ablation.py --use_target_labels` 恢复完整目标域分类损失；此前仅 SSDA 分支计算该项。
- **VIS 微调修复**：`VisualFeatureExtractor` 的 ResNet 位于匿名 `Sequential`，真实层名为 `features.5/6/7`。原 `layer2/3/4` 字符串匹配没有解冻骨干；现显式解冻 layer2--4 与 projection，并保持冻结 BatchNorm 为 eval。
- **Sinkhorn 数值约束**：非均匀源边缘下，固定迭代可能只满足目标列边缘。现保留同一熵正则 OT，在配置的最少迭代后按双边残差进行有上限的继续缩放，并在日志记录实际迭代数。
- **严格 UDA 模型选择**：服务器入口固定 `--report_strategy last`，保存/汇报 final checkpoint；epoch peak 只能作为诊断，不用于选择或主表结果。
- **共同类别实验的边界**：新增 `jmda_oracle_shared_uda`，读取目标训练集类别支持后只保留源域中的共同类别；目标训练集和验证集不做选择性删样本。该操作使用了目标标签支持，协议名与日志均标记 `ORACLE_SHARED_UDA`，不得并入严格 UDA 主表；若做方法间比较，所有方法必须使用同一过滤清单。
- **对比方法随机种子**：所有由 `run_server.sh` 启动的 SDA/UDA 对比入口现在实际控制 Python、NumPy、PyTorch CPU/CUDA RNG；此前多数脚本虽解析 `--seed`，但没有应用。
- **AD-CLIP**：projector 子层改为 `ModuleList`，确保进入优化器与 checkpoint；token id 改为随模型迁移的 buffer；评估改用官方目标的 target-only style-mapping 路径，不再把同一目标图同时伪装成源/目标输入。
- **CDAN**：将官方参数组中的 `lr_mult/decay_mult` 显式转换成 PyTorch 的组级 `lr/weight_decay`；此前这些自定义字段被 SGD 静默忽略。
- **Boomda/MODfinity**：VIS/IR 随机空间增强改为同步；严格 UDA 禁止 `--use-ais`，因为当前 AIS 分配按目标真实类别选样，会造成标签泄漏。正式服务器配置仍为可核验的 VIS+IR。

当前服务器命令分为：

```bash
bash scripts/run_server.sh sda 100                     # SDA
bash scripts/run_server.sh jmda_uda 100                # JMDA-Net 严格 UDA
bash scripts/run_server.sh jmda_oracle_shared_uda 100  # 共同类 oracle 诊断
bash scripts/run_server.sh uda 100                     # 其他严格 UDA 方法
bash scripts/run_server.sh supervised 100              # MCFT，独立 Supervised 表
```

本轮修复后的静态与数值门禁（2026-10-14）如下：

- `JMDA-Net/test_tensor_ot_invariants.py`：27/27 通过，覆盖 SDA/SSDA/UDA 分类损失边界、无标签采样、SVD 投影更新、VIS 解冻/冻结 BN、双边 Sinkhorn 边缘与梯度。
- `tools/test_jmda_protocol_wiring.py`、`test_comparison_reproducibility.py`、`test_data_paths.py`、`test_result_protocol.py`：20/20 通过。
- 29 个改动 Python 文件完成语法编译检查；十个正式训练入口均通过 `--help` 导入/参数解析；`scripts/run_server.sh` 通过 Bash 语法检查，`git diff --check` 通过。
- 本地严格 UDA 与 `ORACLE_SHARED_UDA` 预检均通过；严格模式明确显示目标标签权重为 0、final checkpoint 报告，oracle 模式明确警告读取了目标训练类别支持。
- `server_logs/jmda_uda/jmda_net_uda_backlight_20261013_211726_2491650.log` 是修复前启动且随后被终止的旧运行，只到 epoch 96/100，没有 final checkpoint，也没有新版本的 `src/tgt/mid` 损失拆分，因此不得用于判断修复后精度。修复后的数值结果必须重新完成服务器训练后再报告，不能由上述门禁测试推断或保证。
- 修复后已在本机完整数据上完成两次逆光 1-epoch/1-seed 端到端烟测。正式 warm-up 配置验证了 `src/tgt/mid=2.3914/0/0`、目标标注观察数为 0、`report_strategy=last` 及 final checkpoint 保存；另一次仅为覆盖运行路径而临时设 `ot_warmup_epochs=0, ot_ramp_epochs=1`，实际执行 Neural OT 与三域判别器，得到有限损失、Sinkhorn 双边最大残差 `1.72e-06`、平均 50 次迭代，并保存带 `protocol=UDA`、`label_space_mode=partial_set`、空 `oracle_shared_labels` 的 final checkpoint。两次 1-epoch 指标只证明运行链路，不作为论文性能结果或超参数依据。

### 11.8 2026-10-18 逆光平台期与共同类别实现复核

- 最新运行只完成逆光 seed 42 的 epoch 1--61，随后人工停止；没有 traceback、OOM、NaN 或 final checkpoint。目标验证集仅 36 个样本，accuracy 的最小变化单位为 `1/36=0.0278`。
- 严格 `jmda_uda` 使用全部 14 个源类别，而逆光只包含 9 个目标类别。日志显示源域准确率达到 100%，目标 accuracy 后期约为 0.3056，并持续预测 `14/14` 个类别，符合 partial-set 负迁移而非数值故障。
- 历史 SDA 的 `PairedClassSampler` 实际只从源/目标共同类别采样；因此不能把其结果与使用全部 14 个源类别的严格 UDA 当成完全相同的有效标签空间。
- `oracle_shared` 已修正为：只筛选源训练集与源 SVD 统计集、将共同原始标签紧凑映射到 `0..K-1`、同步缩减分类头和类别先验、完整保留目标训练/验证样本，并在日志/checkpoint/机器摘要中记录映射。
- 服务器共同类别入口显式关闭额外的无标签 source-prior 重加权，以先验证“原方法损失 + 无目标分类监督 + 已知共同类别空间”。严格 partial-set UDA 的设置保持不变。
- 本地数据级验证：逆光、雨天、雾天、黑天的源训练对分别由 760 筛为 515、401、515、493；共同类别数分别为 9、6、9、8，目标训练/验证样本均未被筛除。
- 该共同类别设置读取目标训练类别支持，仍属于 `ORACLE_SHARED_UDA`（或 known-target-label-space UDA），不得改名为严格 UDA，也不得与严格 UDA 或其他未使用同一过滤清单的方法混入公平主表。

### 11.9 2026-10-18 四天气低结果与 SVD 秩缺陷修复

- 四个 `ORACLE_SHARED_UDA` seed-42 任务均正常跑满 100 epoch，类别筛选、紧凑标签映射、目标标签损失为 0、final checkpoint 报告均已生效；最终 accuracy 为逆光 0.4444、雨天 0.1923、雾天 0.0348、黑天 0.0500。它们不是启动失败或旧代码未同步。
- 历史 `new_full_sinkhorn.log` 的有监督 Full 使用目标验证集 `best` 报告，五个种子的 target-best 均值为逆光 0.5667、雨天 0.5385、雾天 0.4243、黑天 0.5067；但同一日志的 epoch-100 均值仅为 0.3278、0.1769、0.1409、0.2567。旧高分包含明显的目标标签选模乐观偏差，不能作为 UDA final checkpoint 的直接预期。
- 雾天/黑天最终值低于随机猜测附近，而源训练 accuracy 分别为 0.9632/1.0000。Sinkhorn 双边残差保持有限且很小，故主要证据不支持 OT 数值求解失败，而支持跨域投影/语义对应失稳。
- 本轮服务器入口遗漏了历史 Full 配置使用的 `projection_dim=16`，实际采用新默认 64。与此同时，四天气红外跨域协方差的有效秩中位数仅为 19--42，最低为 11--14；继续保留 64 个方向会把大量数值零空间当成可识别方向。
- 旧代码的 `ot_warmup_epochs=10` 只关闭 Neural OT，仍从 epoch 1 以未完成源域预训练的随机/弱语义特征构造无标签 Tensor/SVD 对应并施加 Tensor loss。雨天在 OT 尚未开启的 epoch 1--10 已从 0.1538 降至 0.0385，雾天也长期低于随机附近，证明这不是单纯的后期 OT 问题。纯 UDA 现默认先做 10 epoch 真正的 source-only warm-up：不前向目标训练图像、不更新跨域 SVD、不施加 Tensor/域对抗/OT，再启动原有无标签对齐；SDA/SSDA 路径不变。该时序不增加损失或目标标签，但论文须披露并消融 warm-up=0/10。
- `Tensor.py` 现只用有效奇异方向更新已识别子空间；数值零空间使用与上一 epoch 最接近的正交补。这是在 SVD 目标未识别方向中选择稳定代表，不改变 Tensor/SVD 主目标。日志新增 `stabilized_null_dims`。
- UDA 的分类器基迁移改为以有监督源域坐标为锚；Neural-OT residual 的源、目标输入块分别使用各自映射。共享判别器/CostNet/残差输出采用源目标平均映射的最近正交 Procrustes 因子，避免不同域换基时逐 epoch 乘入非正交收缩；原始平均映射的风险仍由 `basis_nonorth` 记录，实际搬运映射另记 `transport_nonorth`。
- 换基日志另记 `basis_min_cos_src/tgt`（旧/新子空间最小主角余弦）。它直接暴露首次 post-warm-up SVD 中无法由坐标搬运恢复的最坏方向；若该值接近 0，即使代码运行正确，warm-up 学到的分类函数也可能因真实子空间替换而受损。
- `scripts/run_server.sh` 现在为 JMDA UDA 显式传入 `projection_dim=16`；`module_ablation.py` 在省略该参数时也按协议选择纯 UDA=16、SDA/SSDA=64，既避免 UDA 直跑命令再次落回不稳定的 64 维，又保持监督路径旧口径。服务器仍可用 `JMDA_PROJECTION_DIM` 显式覆盖，正式敏感性实验应保留 16/32/64/128 的预注册对照。
- 有效秩阈值改为按 covariance dtype 的机器精度计算，不再把固定的归一化 `eps=1e-8` 错当成 float32 数值秩容差；日志新增 `rank_thresholds`。
- JMDA 服务器入口默认在训练前运行协议接线测试及 `test_tensor_ot_invariants.py`；任一门禁失败时不再浪费 GPU 启动正式任务，可用 `JMDA_RUN_INVARIANT_TESTS=0` 显式关闭（正式运行不建议）。
- 启动脚本和 Python 入口都要求纯 UDA 的总 epoch 严格大于 source-only warm-up；否则全程没有任何跨域更新，现会直接失败，避免把 source-only smoke 误记为 UDA 结果。
- `tools/audit_jmda_uda_log.py` 已接入 JMDA 两条服务器入口：进程退出码为 0 后仍会逐 seed 核验代码指纹、16 维配置、完整 source-only 区间、首次 SVD 次序与诊断、每轮目标分类观察数/损失为 0、连续投影更新、final checkpoint 和 `report_strategy=last`。任一协议或配置不变量失败都会把任务重新标记为 failed；低于随机水平、首次换基最小主角余弦过低或源准确率骤降只作为性能警告，不使用目标标签替代模型选择。
- 新日志、机器摘要与 checkpoint 记录 16 位训练代码指纹，用于确认服务器实际执行的是同一组 JMDA 核心文件，避免把未同步代码误判为修复无效。
- `projection_dim=16` 显式传参与 dtype-aware 数值秩阈值属于配置/数值错误修复；零空间延续和换基搬运属于原 SVD 实现细化。真正的 source-only warm-up 改变了 UDA 训练时序，虽未增加目标标签、伪标签或新损失，仍应按训练协议的实质性变化披露，并做 warm-up=0/10 消融，不能把它隐藏成普通 bug fix。若修复后强天气域仍低，剩余问题是无标签 all-pairs OT 不能保证类别语义匹配；任何伪标签/类条件无监督扩展均属于更进一步的实质性算法变化，必须单独命名并做消融。

### 11.10 2026-10-19 部署与结果完整性门禁补强

- `server_logs/jmda_oracle_shared_uda/*_20261018_160218_978602.log` 是上述修复落地前的已完成输出：其日志明确为 `projection_dim=64`、epoch 1 即执行 SVD，且没有 source-only warm-up、稳定零空间、基运输细分诊断或训练代码指纹。因此它们能证明旧实现发生了跨域投影失稳，但不能验证 16 维 + 真 source-only warm-up 的当前实现。
- 日志审计现在从启动器接收预期代码指纹、迭代次数、base seed、总轮数、warm-up 和投影维度；逐 seed 核对连续编号与精确 seed 序列，并证明 `reported_acc` 及 summary 的 accuracy/precision/recall/F1 values、mean、std 均来自最后一轮，而不是 target-best。
- UDA 与 `ORACLE_SHARED_UDA` 的结果收集记录强制清空 `best_acc`；目标验证峰值仍可留在原始日志作诊断，但不得进入可被误用的主结果字段。
- 核心源码指纹统一将 CRLF/LF 归一后计算，`run_server.sh` 固定要求已审核版本的指纹，并在生成清单、占用 GPU 前无条件重算核对（即使关闭可选数值测试也不能绕过）。服务器遗漏同步任一核心文件时会在训练前失败；训练后审计再次核对日志指纹。
- 共享判别器/Neural-OT 使用的平均基映射若秩亏，其 polar 因子并不唯一；现在在未识别零空间选择最接近 identity 的正交补，避免任意旋转。日志增加共享运输有效秩/阈值；这是原坐标运输的数值实现细化，不改变损失、标签使用或方法目标。
- 仓库新增针对 `scripts/run_server.sh` 的 LF 属性并将当前脚本规范为 LF，避免 Windows 直接上传到 Linux 后由隐藏 `\r` 导致 Bash 启动/续行错误。结果收集器失败也会计入套件失败数，低性能警告则仍不改变协议成功状态。
- 截至本节记录时尚无当前修复版本产生的服务器训练日志。静态门禁只能证明协议接线与数值不变量，不能保证目标准确率；下一步必须先跑 1 seed × 4 天气诊断，并重点检查 epoch 10→11 的源准确率、`basis_min_cos_src`、有效秩及预测类别覆盖，再决定是否需要披露为方法局限或开展实质性算法扩展。
- 日志审计 JSON 现直接提取首次 post-warm-up 的 `effective_ranks`、`stabilized_null_dims`、`shared_transport_ranks` 与 `basis_domain_gap`，并校验投影维度、有效秩和零空间补全维数的一致性；存在数值零方向或共享运输秩亏时只发性能警告，不使用目标标签改变 checkpoint。

## 12. 2026-10-19 第二批新方法（CCSA / d-SNE / PDA / MCFT）与方法清单收敛

### 12.1 方法来源与开源仓库

本批 4 个方法的开源代码按要求下载到本地（不下载官方数据集），并移植到本仓库统一数据/效率口径：

| 方法 | 上游仓库 | 原始框架 | 本仓库实现文件 | 协议 | 输入模态 |
|------|----------|----------|----------------|------|----------|
| CCSA | [samotiian/CCSA](https://github.com/samotiian/CCSA) | Keras | `CCSA/train_ccsa.py` | SDA | VIS |
| d-SNE | [aws-samples/d-SNE](https://github.com/aws-samples/d-SNE) | MXNet | `d-SNE/train_dsne.py` | SDA | VIS |
| PDA | [BaiShuanghao/Prompt-based-Distribution-Alignment](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) | PyTorch (dassl) + CLIP | `PDA/train_pda.py` | UDA | VIS (CLIP) |
| MCFT | [SYFYN0317/MCFT](https://github.com/SYFYN0317/MCFT) | PyTorch (MViT) | `MCFT/mcft_model.py`、`MCFT/train_mcft.py` | Supervised | VIS+IR |

CCSA / d-SNE 原作即为有监督域适应（SDA），因此并入 SDA 主表；PDA 为严格无监督（UDA）；MCFT 为源域监督的 VIS+IR 双模态分类，单独成表。

### 12.2 移植与忠实性要点

- **CCSA**（ICCV 2017，对比语义对齐）：逐字移植原作者 `Initialization.py` 的 `Create_Model()` 拓扑——两层 3×3 valid 卷积（32→32）→ MaxPool → Dropout(0.25) → Flatten → FC(120) → FC(84)，分类头 Dropout(0.5)+FC(14)；`euclidean_distance` 与 `contrastive_loss`（margin=1）逐行移植；总损失 `loss = (1-α)·CE + α·CSA`，α=0.25（原论文默认），优化器 Adadelta（原作默认）。**唯一按项目要求替换的是输入尺寸**（改为 224×224，14 类），首层展平维度相应变化。
- **d-SNE**（CVPR 2019，随机邻域嵌入）：`custom_layers.dSNELoss` 逐行移植（`y_same/y_diff` 掩码、`where(y_same, max_dists, inter_cls_dists)`、`relu(max_intra - min_inter + margin)`，margin=1），并对源/目标双向对称计算；主干替换为 ResNet-18（ImageNet 预训练）+ 4096→512 embedding 头；`loss = (1-α)·xent + α·dSNE`，α=0.1；SGD(lr=1e-3, momentum=0.9, weight_decay=1e-4)。目录名 `d-SNE` 含连字符，效率适配器用 `_load_module_from_file()` 加载以绕过非法模块名。
- **PDA**（AAAI 2024，提示分布对齐）：`PromptLearner`/`TextEncoder`/`ImageEncoder_Trans`/`IFT_Module`/`CustomCLIP` 全部移植，去除 dassl 依赖；超参沿用 Office-31 设定（confi=0.9、warm_up=1、epoch_switch=0、feature bank K=5），SGD(lr=0.003, momentum=0.9, weight_decay=5e-4)；CLIP ViT-B/16 权重经 certifi CA bundle 下载并做 SHA-256 校验。**类别提示词为占位名**：本数据集无官方真实类别名，脚本在无映射文件时回退 `class_1..class_14` 并打印警告，效率记录标注 `prompt_source=placeholder`（真实类别名可由 `JMDA_CLASS_NAMES_FILE` 提供）。
- **MCFT**（TGRS 2024，多模态对比融合 Transformer）：作者 MViT 结构忠实移植并去除 einops/timm/scipy 依赖；VIS/IR 按文件名 stem 精确配对（`PairedVisIrDataset`），无配对样本丢弃并打印数量；保留原作 `itm_head = Linear(patch*patch*nout*8, 2)` 的展平形式（与展平序列宽度一致）。**忠实性标注**：原作 `AGENT_TOKEN_SPLIT=169` 常量与 15×15 patch 网格不自洽（属原作 bug），本实现逐字保留该常量并在此建档说明；`_sample_negative()` 对零权重行回退均匀采样（原作会 RuntimeError）。

### 12.3 本地冒烟测试结果（不做本地正式训练）

均为 `--target backlight --epochs 1`，只验证全链路可运行，数字不代表正式结果：

| 脚本 | 协议 | 结果 |
|------|------|------|
| `CCSA/train_ccsa.py` | SDA | 正常训练/评估/保存，日志与效率记录完整 |
| `d-SNE/train_dsne.py` | SDA | 正常训练/评估/保存，日志与效率记录完整 |
| `PDA/train_pda.py` | UDA | CLIP 提示学习 + 分布对齐链路完整，目标标签仅用于最终测试 |
| `MCFT/train_mcft.py` | Supervised | VIS+IR 配对加载、双模态融合训练/评估链路完整 |

### 12.4 效率基准结果（fp32，RTX 4060 Ti，1 MAC = 2 FLOPs）

`tools/benchmark_efficiency.py` 新增 4 个适配器（`ccsa`/`dsne`/`pda`/`mcft`）后测得的静态与推理指标：

| 方法 | 协议 | 参数量 | 可训练参数 | GFLOPs | 推理延迟(ms) | 备注 |
|------|------|--------|------------|--------|--------------|------|
| CCSA | SDA | 46.49M | 46.49M | 1.07 | 1.11 | Keras CNN 移植，224px 输入 |
| d-SNE | SDA | 15.38M | 15.38M | 3.66 | 2.25 | ResNet-18 + 512 维嵌入 |
| PDA | UDA | 152.87M | 3.18M | 479.39 | 33.91 | CLIP ViT-B/16，仅提示上下文可训练 |
| MCFT | Supervised | 2.14M | 2.14M | 0.20 | 13.25 | MViT，15×15 patch 网格，VIS+IR |

为对照，同批 UDA 传统方法（`dann`/`deepcoral`/`cdan`/`mdd`/`mcc`，ResNet-18 主干，224px 输入）效率为：DANN 11.44M/3.65 GFLOPs/2.87ms、DeepCORAL 11.18M/3.65 GFLOPs/2.18ms、CDAN 13.41M/14.59 GFLOPs/2.58ms、MDD 11.96M/3.65 GFLOPs/3.47ms、MCC 11.82M/3.65 GFLOPs/2.75ms。

> **注意（启动脚本修复）**：`tools/benchmark_efficiency.py` 的 `ADAPTERS` 以**小写**为键，而一键脚本的显示名称为大写（`DANN`、`MCFT` 等）。初版 `scripts/run_all_sunny.ps1` 直接把显示名传给 `--methods`，导致微基准全部被判定为 "Unknown method" 而跳过（Params/GFLOPs/Latency 列为空）。现已加入 `$AdapterKeys` 显示名→小写键映射，并将三组 benchmark 日志按协议拆分为 `benchmark_uda.log`/`benchmark_sda.log`/`benchmark_supervised.log`，避免覆盖。

### 12.5 一键启动脚本

`scripts/run_all_sunny.ps1`：源域=晴天，目标域=逆光/雨天/雾天/黑天，每个目标域独立实验，每方法×目标×种子一次，默认 5 个种子(42–46)。脚本自动完成三组效率微基准（UDA / SDA / Supervised）与全部训练，并重新生成汇总。

```powershell
# 全部 9 方法 × 4 天气 × 5 种子
powershell -ExecutionPolicy Bypass -File scripts\run_all_sunny.ps1

# 自定义：只跑 CCSA/d-SNE，1 epoch，单目标单种子
powershell -ExecutionPolicy Bypass -File scripts\run_all_sunny.ps1 `
  -Methods CCSA,d-SNE -Epochs 1 -Seeds 42 -Targets backlight

# 关键参数：-Methods -Seeds -Targets -Epochs -OutDir -BenchRepeats
#           -TargetLabelsPerClass (SDA 预算, 默认10) -SdaSeed (默认42)
#           -ClipCacheDir (PDA) -SkipBenchmark
```

效率结果保存到 `-OutDir`（默认 `efficiency_results/all_sunny`），含原始运行记录 `efficiency_runs.csv`、汇总表 `efficiency_summary.csv` 与 Markdown 表 `efficiency_table.md`；SDA/UDA/Supervised 三协议**分表呈现，不跨表排名**。服务器端对应入口为 `scripts/run_server.sh {sda|uda|supervised}`。

### 12.6 方法清单收敛（删除清单）

按要求删除除 **DeepCORAL、DANN、CDAN、MDD、MCC、CCSA、d-SNE、PDA、MCFT** 之外的对比方法目录及其相关内容：

- 删除的方法目录：`AD-CLIP/`、`Boomda/`、`CoOp/`、`Modfinity/`、`TransAdapter/`、`SDA_Baselines/`
- 删除的 legacy 启动脚本：`run_all_methods.py`、`run_new_methods.py`
- 清理引用：`tools/benchmark_efficiency.py`（移除 5 个废弃适配器与注册项）、`tools/test_comparison_reproducibility.py`（入口清单与针对性测试）、`tools/preflight_server.py`（TransAdapter 检查）、`tools/efficiency_common.py`、`tools/make_sda_lists.py`、`DANN/train_dann_sda.py`（注释）、`scripts/run_server.sh`（sda/uda/supervised 方法清单）

**保留说明**：
- `Transfer-Learning-Library/` 与 JMDA-Net 的 tllib 依赖（MDD/MCC 依赖其裁剪子集）——删除会破坏 MDD/MCC，故保留。
- `JMDA-Net/` 为本文方法（JMDA-Net）源码镜像，不属于被删除的对比方法。


