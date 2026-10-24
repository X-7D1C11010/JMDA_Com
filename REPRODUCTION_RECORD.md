# 域适应方法在 JMDA-Net 数据集上的复现操作记录

VDS2Raw v2 的独立三模态跨日协议、训练入口和输出说明见
[`VDS2RAW_PROTOCOL.md`](VDS2RAW_PROTOCOL.md)。

## CCSA / d-SNE / MCFT 顺序训练入口

三种官方方法移植版可以用一个命令按 `CCSA -> d-SNE -> MCFT` 的顺序训练、
记录并汇总：

```bash
export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data
bash scripts/run_ccsa_dsne_mcft_background.sh formal
```

将 `formal` 换成 `smoke` 可执行“一轮、一批次”的端到端预检。可通过
`JMDA_TARGETS` 选择目标天气，通过 `JMDA_SEEDS` 指定训练随机种子。默认源域为
晴天、目标域为全部四种天气、训练种子为 42；CCSA 和 d-SNE 固定共享 10-shot、
seed 42 的有标签目标子集。每个运行目录保存原始日志、模型、逐轮指标 CSV/JSON、
效率记录、总表 `results/runs.csv` 以及按协议生成的 CSV/Markdown 汇总。三种方法
均不需要真实类别名称文件 `class.txt`。

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

## 13. 2026-10-20 训练曲线诊断与保真性审计

针对"5 个 UDA 方法都呈'先升后降/进入平台'的相同趋势、是否复现不完全"的疑问，做两项工作：(1) 把逐 epoch 指标从日志导出成 CSV 与曲线；(2) 把各方法本仓库实现与**仓库内留存的原参考实现**逐条对照。结论：**该趋势主要由"小数据 + 小验证集"引起的源域过拟合造成，不是复现失败；但 DANN/DeepCORAL/CDAN 确实存在真实的训练配方偏差，其中 CDAN 的偏差最可能放大后期退化。**

### 13.1 逐 epoch 曲线导出工具

新增 `tools/extract_epoch_curves.py`（**纯后处理**，只解析训练脚本已写出的日志，不触碰任何方法的训练代码、超参或 checkpoint）：

```
python tools/extract_epoch_curves.py \
  --logs-dir efficiency_results/uda_sunny/logs \
  --out-dir  efficiency_results/uda_sunny/epoch_curves
```

产出（`efficiency_results/uda_sunny/epoch_curves/`）：`curves/<方法>_<目标域>_seed<种子>.csv`（100 份，逐 epoch 准确率/P/R/F1/覆盖度/各损失项均值）、`all_epochs.csv`（合并长表）、`run_peak_final.csv`（每次运行的峰值/末期/差值）、`plots/<方法>_<目标域>.png`（20 张，多种子均值±标准差，含准确率与损失双面板）。

**注（编码）**：`efficiency_results/` 下的日志由 PowerShell 5.1 的 `>` 重定向写出，编码为 **UTF-16LE(BOM)**，任何按 UTF-8 读取的脚本都会解析失败；工具已按 BOM 自动判定编码。汇总表 `efficiency_table.md` / `efficiency_summary.csv` 则是 GBK，非 UTF-8。

### 13.2 趋势的量化证据

从 100 个 run 的逐 epoch 曲线聚合（`peak_epoch` = 目标域 val 准确率最高的轮次）：

| 方法 | 平均峰值轮次 | 峰值准确率 | 末期准确率 | 峰值−末期 | 峰值出现在第 10 轮之后的 run 数 |
|------|------------|-----------|-----------|----------|------------------------------|
| DANN | 6.4 | 0.3624 | 0.3108 | 0.0516 | 4/20 |
| DeepCORAL | 6.5 | 0.3313 | 0.2841 | 0.0472 | 5/20 |
| CDAN | 10.2 | 0.3464 | 0.2767 | 0.0696 | 10/20 |
| MDD | 10.6 | 0.3571 | 0.2686 | 0.0885 | 12/20 |
| MCC | 9.2 | 0.4288 | 0.3747 | 0.0541 | 7/20 |

- **退化是真实的**：峰值普遍出现在第 6–13 轮，之后到第 20 轮平均掉 4.7–8.9 个点；个例更大（CDAN 雨天 0.262→0.154，掉 10.8 点；MDD 黑天 0.198→0.092；MDD 逆光 0.528→0.422）。
- **根因是源域过拟合，不是欠拟合**：源域训练损失在 5–10 轮内被压到接近 0（DANN `loss_cls`→0.05~0.13；CDAN `loss_cls`→0.0012）。源域训练集仅 986 张 VIS，配 ImageNet 预训练 ResNet-18，很快被完全记住；此后模型趋于"源域专用"，域不变性下降。若是未收敛，源域损失应仍很高。
- **因此加 epoch 会加重而非缓解该趋势**："增加训练轮数"不是该趋势的解法。
- **100 轮的直接佐证**：历史有监督 Full（`new_full_sinkhorn.log`）跑到第 100 轮的 epoch-100 均值（逆光 0.3278 / 雨天 0.1769 / 雾天 0.1409 / 黑天 0.2567）远低于同批的 target-best 均值（0.5667 / 0.5385 / 0.4243 / 0.5067），说明轮数更长时退化幅度更大，与"过拟合"判断一致。
- **量化噪声不可忽视**：目标域 val 仅 36（逆光）/26（雨天）/115（雾天）/120（黑天）个样本，单样本分别等价 2.78%/3.85%/0.87%/0.83%。逆光/雨天逐轮 1–3 个样本的抖动即表现为"范围内波动"，不应据此判断方法优劣；雾天/黑天粒度细，其平台更可信。
- **CDAN 的对抗对齐确实在恶化**：`loss_transfer` 从 0.09 升到 0.84，同时 `loss_cls`→0.001，属"判别器压制特征"的典型退化。

### 13.3 保真性审计（逐条对照仓库内留存的原参考实现）

| 方法 | 项目 | 原参考实现（仓库内留存） | 本仓库实现 | 判定 |
|------|------|------------------------|-----------|------|
| DANN | 优化器 | **Adam**(lr=1e-3)　`DANN/train/main.py:76` | **SGD**(lr=1e-3, mom=0.9, wd=1e-4)　`DANN/train_dann.py:168` | **偏差** |
| DANN | batch / epochs | 128 / 100　`main.py:21,23` | 32 / 20 | 偏差（随数据规模调整） |
| DANN | alpha 调度 | α=2/(1+e^(−10p))−1　`main.py:100-101` | 同 | 一致 |
| DANN | LR 调度 | 无 | 无 | 一致 |
| DeepCORAL | 优化器参数组 | 2 组：主干=LR、**分类头 fc=10×LR**　`DeepCORAL/main.py:130-133` | 单组统一 LR　`train_deepcoral.py:148` | **偏差** |
| DeepCORAL | CORAL 权重 | **λ=(e+1)/EPOCHS 线性递增**　`main.py:148` | **固定** `args.lamb`　`train_deepcoral.py:194` | **偏差** |
| DeepCORAL | weight_decay | 定义了 5e-4 但未传入优化器 | 1e-4 | 偏差（小） |
| DeepCORAL | EPOCHS / lr | 20 / 1e-3 | 20 / 1e-3 | 一致 |
| CDAN | 优化器 | SGD(**nesterov=True**, wd=**5e-4**)　`pytorch/train_image.py:244-245` | SGD(wd=1e-4, 无 nesterov)　`train_cdan.py:144` | **偏差** |
| CDAN | LR 调度 | **`inv`：lr·(1+0.001·iter)^(−0.75)，逐迭代**　`pytorch/lr_schedule.py` + `train_image.py:166,246` | 常量 LR（无调度） | **偏差** |
| CDAN | coeff | `calc_coeff(iter)` | `calc_coeff(iter_num)` | 一致 |
| CDAN | batch | 36 | 32 | 偏差（小） |
| MDD | 全部 | tllib `examples/.../mdd.py`（lr 0.004 / γ 2e-4 / decay 0.75 / nesterov / wd 5e-4 / batch 32） | 完全一致　`MDD/train_mdd.py:200-202` | 一致 |
| MCC | 全部 | tllib `examples/.../mcc.py`（lr 0.005 / γ 1e-3 / decay 0.75 / nesterov / wd 1e-3 / temp 2.5 / batch 36） | 完全一致　`MCC/train_mcc.py:193-194` | 一致 |

**共同的有意适配**（第 6 节已记录，不属保真性偏差）：主干网络统一替换为 ResNet-18（原 DANN 为 MNIST-CNN、DeepCORAL 为 AlexNet@300×300、CDAN 为 ResNet-50、tllib 示例默认 ResNet-50），输入 224×224、14 类。

**审计边界**：`DANN/` 内留存的是 fungtion/DANN 的 MNIST 版 `train/main.py`（Adam）；该仓库其它变体/DANN 论文原始 Theano 实现是否使用 SGD 未在本仓库留存、无法逐行核对，故上表 DANN 一行仅相对**本地留存版本**判定。MDD/MCC 可与 tllib 官方示例逐行核对，结论可靠。

### 13.4 结论与建议

1. **"所有方法同一趋势"的解释**：它们共享同一源域（986 张 VIS）、同一 20 轮预算与同样偏小的目标域验证集，因此共同呈现"源域快速过拟合 → 目标域先升后稳/微降"。这与各方法是否有独立 bug 无关。
2. **有真实偏差的是 3 个**：DANN（Adam→SGD）、DeepCORAL（丢失分类头 10×LR、丢失 λ 线性递增）、CDAN（丢失 `inv` LR 衰减、nesterov、wd 5e-4）。其中 **CDAN 的偏差与其实测最严重的后期退化（雨天 −10.8 点）方向一致**，最可能是放大因素；但 MDD 配置完全忠实却仍有最大峰值−末期差（0.0885），说明**主因仍是过拟合/数据规模，而非配方偏差**。
3. **若要归因到底**：这些偏差都是可检验的——补做对照实验（如给 CDAN 加回 `inv` 调度、给 DeepCORAL 加回 λ 递增与 10× 分类头 LR）即可分离"配方偏差"与"数据规模"两因素的贡献。该对照会改变优化过程，按项目约束需单独命名并披露，且需用户授权后进行。
4. **不建议的做法**：单纯增加 epoch。源域损失已达 ≈0，延长训练只会加重过拟合。

## 14. 2026-10-20 UDA 数据口径变更与训练配方偏差修复

依据第 13.3 节审计结论，并经用户授权，对 6 个无监督域适应（UDA）方法做两类改动：**(A) 数据口径改为"以目标域类别为准"；(B) 修复 3 处训练配方偏差**。本节记录改动内容与影响范围。

### 14.1 数据口径变更：标签空间以目标域为准

**问题**：晴天（源域）含全部 14 类，而各目标域天气场景只含其中一部分类别。按原先的 14 类实现，源域里存在大量"目标域根本不存在的类别"的图像，无监督对齐会被迫把这些无法被目标域解释的类簇也压到一起，且分类头里有相当比例的类别在目标域上永远不可能被预测正确，准确率分母被稀释。

**新口径**（用户给定）：
- 目标域类别集 `T` = 该目标域 train ∪ val 列表中出现的原始类别 id；
- 源域 **train** 与目标域 train/val 都只保留 `label ∈ T` 的图像；
- 标签重映射到 `0..|T|-1`，**分类头输出维度 = |T|**；
- 因此随机基线由全局 1/14=7.14% 变为 1/|T|。

**公共实现**：新增 `tools/uda_data.py`（无监督方法共用的唯一数据入口），提供 `uda_label_space(data_dir, target)`→`(class_ids, label_map)`、`UdaImageListDataset(list_path, transform, label_map)`、`format_class_space` / `format_dropped`（日志用）。每个训练脚本启动时打印类别集、保留/丢弃数量。

**过滤后的实际样本量与随机基线**（`data_lists/`，源域只过滤 `sunny_train.txt`，因为 6 个 UDA 脚本不使用源域 val）：

| 目标域 | |T| | 原始类别 id | 目标 train / val | 源域 train 保留 / 丢弃 | 随机基线 |
|--------|-----|-------------|-----------------|----------------------|---------|
| 逆光 backlight | 9 | 0,4,5,6,7,8,10,12,13 | 128 / 36 | 643 / 343 | 11.1% |
| 雨天 rainy | 6 | 0,4,5,6,8,10 | 96 / 26 | 525 / 461 | 16.7% |
| 雾天 foggy | 9 | 0,1,3,4,5,7,8,9,10 | 440 / 115 | 713 / 273 | 11.1% |
| 黑天 dark | 8 | 0,1,4,5,6,8,10,11 | 465 / 120 | 629 / 357 | 12.5% |

**接入范围**：`DANN/train_dann.py`、`DeepCORAL/train_deepcoral.py`、`CDAN/train_cdan.py`、`MDD/train_mdd.py`、`MCC/train_mcc.py`、`PDA/train_pda.py`（这 6 个是 UDA 协议）。每处原有的本地 `ImageListDataset` 类已删除，改用 `uda_data` 的类；`classification_metrics(...)` 调用改为显式传入 `num_classes=|T|`（否则默认 14 的范围校验会拒绝）。PDA 的 CLIP 占位类名由 `class_1..class_14` 改为 `class_1..class_{|T|}`。

**当时未接入**：`CCSA`、`d-SNE`、`MCFT` 与各 `*_sda.py` 变体。CCSA、d-SNE、MCFT 后续已在第17节接入同一目标驱动标签空间；本句只描述第14节改动发生时的状态。

**可比性提醒**：过滤后各目标域的类别数不同（6/8/9/9），因此**跨目标域的准确率绝对值不能直接比较**，随机基线也不同；与第 8 节、第 13.2 节的历史数值不可直接对比。

### 14.2 训练配方偏差修复（仅限第 13.3 节审计出的 3 处）

| 方法 | 修复前 | 修复后（对齐仓库内原参考实现） |
|------|--------|-------------------------------|
| DANN | `SGD(lr=1e-3, momentum=0.9, weight_decay=1e-4)` | **`Adam(lr=1e-3)`**　（`DANN/train/main.py:76`，原实现不传 momentum/weight_decay） |
| DeepCORAL | 单参数组统一 LR；CORAL 权重固定 `--lamb`；`weight_decay=1e-4` | **2 参数组：`sharedNet`=lr、`fc`=10×lr**（`main.py:130-133`）；**λ=(epoch+1)/EPOCHS 线性递增**（`main.py:148`）；**不传 weight_decay**（原实现定义 5e-4 但未传给优化器）。原 `--lamb` 参数已移除 |
| CDAN | `SGD(wd=1e-4)`（无 nesterov、无 LR 调度），参数组经 `optimizer_utils.materialize_cdan_parameter_groups` 预展开 | **`SGD(momentum=0.9, weight_decay=5e-4, nesterov=True)`** 且直接使用 `get_parameters()` 的 `lr_mult`/`decay_mult` 分组；**逐迭代**调用 `inv_lr_scheduler`：`lr = lr·(1+0.001·iter)^(−0.75)`，`weight_decay = 5e-4·decay_mult`（`pytorch/train_image.py:166,244-246` + `pytorch/lr_schedule.py`）。新增 `--lr-gamma`(0.001) / `--lr-decay`(0.75) 两个可调参数，默认即原实现值 |

`CDAN/optimizer_utils.py` 保留（仍被 `CDAN/train_cdan_sda.py` 与 `tools/test_comparison_reproducibility.py` 使用），只是不再被 UDA 版 `train_cdan.py` 引用。

**未改动**：`MDD`/`MCC` 与 tllib 官方示例逐行一致，本次一处未动；所有方法的网络结构、损失函数、alpha/temperature 等超参、随机种子逻辑、效率日志逻辑均未改动。除上述 3 处外，其余"有意适配"（主干换 ResNet-18、输入 224×224 等）保持不变。

### 14.3 效率基准的口径说明

`tools/benchmark_efficiency.py` 新增 `--num-classes`（默认 14）以设置各方法分类头宽度，并把实际取值写入每次运行的 `benchmark.log`。该基准不加载任何真实数据（分类标签为随机张量，符合"效率测试不使用目标域真实标签"的约束），历史行仍按 14 类生成；UDA 训练改为 |T| 类后，分类头宽度差异对 ResNet-18 的参数量/FLOPs 影响 < 0.1%（如 512×(14−9)=2560 参数 vs 约 11.2M）。如需与训练严格对齐，可在基准命令上加 `--num-classes <|T|>`。

### 14.4 验证与重跑

1. **冒烟验证已通过**（`--target backlight --epochs 1 --seed 42`）：DANN / DeepCORAL / CDAN / MDD / MCC / PDA 六个方法全部 exit=0，日志确认：
   - `UDA label space (target-driven): 9 classes, original ids [0, 4, 5, 6, 7, 8, 10, 12, 13], remapped to 0..8`
   - `source train: kept 643, dropped 343`
   - DeepCORAL `optimizer group learning rates: [0.001, 0.01]`（2 参数组生效）
   - CDAN `optimizer group lr_mult: [1, 10, 10, 10]`，且 batch 日志中 lr 逐迭代递减（`9.93e-04 → 9.85e-04`），inv 调度生效
   - 冒烟产物在系统临时目录，未写入 `efficiency_results/`。
2. **当前重跑配置**（用户指定）：6 个 UDA 方法 × 目标域 backlight × 单种子 42 × **100 epoch**，输出目录 `efficiency_results/uda_sunny_targetspace/`：
   ```
   powershell -ExecutionPolicy Bypass -File scripts\run_all_sunny.ps1 `
       -Methods DANN,DeepCORAL,CDAN,MDD,MCC,PDA `
       -Targets backlight -Seeds 42 -Epochs 100 `
       -OutDir efficiency_results\uda_sunny_targetspace
   ```
   该脚本自动完成效率微基准 + 训练 + 汇总表重生成；逐 epoch 曲线用
   `tools/extract_epoch_curves.py` 从 `logs/` 另行导出。
3. 输出目录必须新建（如上例），**不要**复用 `efficiency_results/uda_sunny/`——第 8/13 节的历史结果保留原地，两者口径不同，不可混算。
4. 本轮只覆盖 backlight 一个目标域，其余三个目标域（rainy/foggy/dark）的旧结果仍是 14 类口径，**不可与本轮数值并列比较**；若需完整结论，应补齐 `-Targets rainy,foggy,dark` 与 `-Seeds 42,43,44,45,46`。

### 14.5 100 epoch 重跑结果（backlight × seed42，新口径）

6 方法全部 exit=0。逐 epoch 曲线见 `efficiency_results/uda_sunny_targetspace/epoch_curves/`（curves/all_epochs.csv、run_peak_final.csv、plots/）。

| 方法 | 首轮 | 峰值轮次 | 峰值 | 末轮(epoch100) | 峰值−末轮 | 旧口径对照(20轮,14类)：峰值轮/峰值−末轮 |
|------|------|---------|------|----------------|-----------|----------------------------------------|
| PDA | 0.1389 | 26 | **0.8056** | 0.7222 | 0.0834 | （旧批次无 PDA 行） |
| MDD | 0.2500 | 36 | 0.6667 | **0.6667** | **0.0000** | 13 / 0.0834 |
| MCC | 0.4444 | 13 | 0.6667 | 0.6389 | 0.0278 | 9 / 0.0555 |
| DeepCORAL | 0.5833 | 8 | 0.6944 | 0.6667 | 0.0277 | 6 / 0.0277 |
| DANN | 0.4444 | 4 | 0.5833 | 0.4722 | 0.1111 | 3 / 0.0278 |
| CDAN | 0.6667 | 1 | 0.6667 | 0.5000 | 0.1667 | 3 / 0.1112 |

（随机基线不同：新口径 1/9=11.1%，旧口径 1/14=7.1%，绝对值不可直接跨口径比较；表内对照列仅用于看"峰值位置与衰减幅度"两个趋势量。）

**结论**：
1. **此前"初期上升后波动/下降"的主因确认是口径+配方问题，且大部分已消除**：MDD 峰值−末轮从 0.083 → **0.000**（100 epoch 全程稳定在峰值），MCC 从 0.056 → 0.028，DeepCORAL 维持 0.028。三者新口径下 100 epoch 曲线为"上升→高位平台"，不再衰减。
2. **PDA 是唯一持续爬升的方法**（0.14 → 峰值 0.81），2 个 token 的可学习提示 + 伪标签在 26 轮才到峰，说明此前 20 epoch 截断对它明显不够。
3. **DANN/CDAN 仍呈"早峰回落"**，但这是对齐原参考配方后的**忠实行为**而非复现缺陷：
   - CDAN 第 1 轮即 0.667（inv 调度初期 lr 高、nesterov 收敛快），随后对抗训练把判别特征磨平，~20 轮后在 0.47–0.50 平台**波动**（40/60/80/100 轮 = 0.472/0.472/0.500/0.500），并非持续崩塌；
   - DANN（Adam、无 wd）60 轮处最低 0.333，80–100 轮回升至 0.44–0.50，呈长周期振荡。这与 DANN 原论文中对抗训练不稳定的现象一致。
   - 两者如需报告更优数值，惯例是取**目标域 best epoch 的诊断值**（CDAN 0.667、DANN 0.583），但按项目约束不得用目标域真实标签选模型，正式汇报仍用末轮值。
4. **新口径整体抬升了绝对精度**（类别数从 14 → 9，且去掉了目标域不存在的类的干扰）：同方法同种子下峰值普遍 +0.05~0.11。
5. 效率微基准本轮未传 `--num-classes`，仍按 14 类头计算（影响 <0.1%，见 14.3）；`efficiency_table.md` 中 CDAN GFLOPs=14.59 系其 bottleneck+多层对抗网络所致，属方法本身结构差异。

### 14.6 2026-10-21 数据完整性核验、曲线工具修复与第 4 处偏差修复

用户对结果偏低、曲线异常提出质疑后做了三项核查（全部只读或纯后处理，不改训练结果）：

**A. 数据加载完整性核验（结论：数据无误）**
- 全部 14 份列表逐行校验：**label = 类别目录号 − 1** 对源/目标域全部成立（目录名为纯数字 `1..14`，非 `class_x`）。
- 目标域"缺失类"在磁盘上**确实是空目录**（train/val 一致）：逆光缺 {2,3,4,10,12}、雨天缺 {2,3,4,8,10,12,13,14}、雾天缺 {3,7,12,13,14}、黑天缺 {3,4,8,10,13,14}（目录号），"以目标域类别为准"的前提成立。
- **模态类别对应**：4 个目标域 train/val 的 VIS 与 IR 非空类别集合**完全一一对应**（逐类图像数可不同，如雾天 train 类 11 VIS=74/IR=119）；data_lists 仅收录 VIS（各方法按原始设定单模态 VIS 训练）。
- 目标域类别集 ⊆ 源域类别集，label_map 为同一映射对象，重映射一致性成立。

**B. 曲线提取工具修复（纯后处理）** `tools/extract_epoch_curves.py`
- 正则原来只匹配 tllib 格式 `iter [i/n]`，DANN/DeepCORAL/CDAN/PDA 的 `batch [i/n]` 全部漏配 → 这就是"图中 cls 之外三条线基本没有"的原因；已改为 `(?:iter|batch)`，并让数值支持科学计数法（`lr=9.93e-04`）。
- `src_acc`/`acc_x`（0-100 量纲）不再画进 loss 面板（此前会把损失线压扁）。6 份新日志已重新解析出图，所有方法的 loss/lr/alpha/lambda 曲线现已齐全。
- 说明：图中 "cls" 线来自 MDD/MCC 日志的源域分类损失，8-9 类下初始值 ≈ ln|T| ≈ 2.1-2.2，且与对抗/自训练损失耦合，不呈单调幂函数衰减属正常现象；batch 级日志每 epoch 仅 2 个采样点（batch 10/20），曲线天然粗糙。

**C. 第 4 处配方偏差修复**：CDAN 原参考 `train_image.py:198` 的方法默认值为 **`CDAN+E`**（熵条件化，论文最优变体），此前脚本默认 `CDAN`。已改默认为 `CDAN+E`（E 分支本就按原版实现，仅默认值对齐）。**因此上表 CDAN 行是无熵条件化版本的结果**；如需与论文口径一致，需用新默认重跑 CDAN。

**D. "结果偏低 + 峰值集中在前 50 轮"的量化解释：迭代量只有官方协议的约 1/10**
- 本项目 epoch = max(⌈643/32⌉-1, …) = **20 个 batch**（源域 drop_last 后 20 批），100 epoch = **2000 次迭代**。
- tllib 官方示例为 1000 iters/epoch × 20 epoch = **20000 次迭代**；DeepCORAL 原参考 office31 上 batch=200/56、EPOCHS=20（≈上万迭代）。
- 后果：源域分类损失第 2 轮即 ≈0（过拟合进入对抗博弈阶段），且 inv 调度远未退火——MDD γ=0.0002 在 2000 迭代处 lr 仅衰减到 78%（官方 20000 迭代处为 30%），CDAN γ=0.001 仅到 44%（官方 12%）。
- 叠加目标域 val 仅 26-120 张（逆光 36 张，1 张=2.8%），逐轮精度噪声大。
- **建议**：把训练时长对齐官方迭代量级（如 `-Epochs 1000` ≈ 20000 iters，或按 GPU 预算取 300-500），预期峰值位置会后移、末轮值更接近原方法报告水平；这是运行配置调整，不属于方法超参改动。

### 14.7 2026-10-21 1000 epoch 中止 + 低分归因（checkpoint 推理诊断）

**A. 1000 epoch 重跑已中止（非崩溃）**：DANN 停在 epoch 912/1000，target val = 0.4444；从 epoch ≈400 到 912 共 500+ 轮无任何提升，100 epoch 版本同样为 0.4444。**延长训练不能解决 DANN 的问题**，而 1000 epoch × 6 方法按实测吞吐需 ~46 小时（DANN 28.6 s/epoch，数据加载占 ~95%，GPU 利用率仅 2%）。产物在 `efficiency_results/uda_sunny_targetspace_e1000/`（仅 DANN 日志与 benchmark，无 checkpoint）。

**B. 数据泄漏核查（无泄漏）**：按"类别目录 + 文件名"比对，5 个域的 train/val 重复数均为 **0**（源域 986/255、逆光 128/36、雨天 96/26、雾天 440/115、黑天 465/120）。目标域 val 未参与训练。

**C. checkpoint 推理诊断（逆光 val，n=36，9 类，随机基线 0.2222）**

| 方法 | 源域 val | 目标域 val | 域差 | target top-1 | target top-3 | top3−top1 |
|------|---------|-----------|------|-------------|-------------|-----------|
| DANN | 0.9939 (n=165) | 0.4722 | 0.522 | 0.472 | 0.722 | 0.250 |
| DeepCORAL | 1.0000 | 0.6667 | 0.333 | 0.667 | **0.972** | 0.306 |
| CDAN | 1.0000 | 0.5000 | 0.500 | 0.500 | 0.889 | 0.389 |
| MDD | 0.9758 | 0.6667 | 0.309 | 0.667 | 0.806 | 0.139 |
| MCC | 0.9636 | 0.6667 | 0.297 | 0.667 | 0.889 | 0.222 |

逐类召回（新标签 0..8）：类 2、类 8 在所有方法上均 1.00；类 7 在所有方法上均为 0.25；类 1、类 4、类 6 多数方法为 0.00。DeepCORAL 的 top-3 达 35/36。

**判定**：模型对**源域已完全记忆**（96–100%），但对目标域只有一半正确；而 **top-3 高达 0.72–0.97**，说明"表征基本可用、最终决策边界混淆"，不是特征完全失效。低分不是数据口径或代码错误，而是**在 643 张源域图上全量微调 11M 参数的预训练 ResNet-18** 造成的源域过拟合 + 9 类决策头过拟合。

**D. 训练动力学证据**
- 源域分类损失降到 <0.01 的轮次：CDAN 第 2 轮、DeepCORAL 第 6 轮、DANN 第 45 轮、MDD 第 49 轮 → 分类头在极早期即完成记忆化。
- DANN 的 `loss_domain` 全程 ≈1.31–1.45（=2·ln2=1.386 为二分类随机水平）→ 域判别器长期处于"猜不出"，对抗信号饱和。
- CDAN 的 `loss_transfer` 第 5 轮降到 0.095 后又回升到 0.70 → 对抗项先被判别器"打穿"再反弹，属对抗不平衡。
- DeepCORAL 的 `loss_coral` 全程打印 0.0000：原实现 `loss/(4*d*d)`，d=512 时除以 1048576，损失量级 ~1e-6，相对 CE(~1e-3) **数值上被淹没**，对齐项实际近乎失效（该除数与原仓库一致，不是移植错误）。
- PDA 是唯一例外：冻结 CLIP ViT-B/16、仅 3.18M 可训练提示 → 峰值 0.806、末轮 0.722、后 20 轮 std=0.0000。**"冻结强预训练特征 + 轻量适配"明显优于"全量微调 ResNet-18"**，这是本轮最重要的对照。

**E. 待决策的方向（均需用户确认，因为会改变"原配方"）**
1. **统一使用 tllib 标准优化配方**（SGD backbone lr=0.001、头 lr=0.01、wd=5e-4、nesterov，即 MDD/MCC 当前配方）到 DANN/DeepCORAL/CDAN。依据：MDD/MCC 是本次唯二"峰值后不衰减"的方法；而 DANN 的 `Adam(lr=1e-3)`、CDAN 的 `SGD(lr=1e-3)` 是为**小网络从头训练**设计的，与"预训练 ResNet-18 + 643 张图"不匹配，属于"换了骨干却沿用原学习率"的移植瑕疵。
2. **冻结骨干、只训练分类头/瓶颈**（改动更大，但可直接验证过拟合假设）。
3. **仅报告 top-1 末轮并以 top-3 作为辅助指标**，同时增加种子数（逆光 val 仅 36 张，单张即 2.8%，5 个种子的标准差不可忽略）。
4. 补一个 **source-only 基线**（晴天训练、逆光测试）：若其目标域准确率也接近 0.55–0.60，则说明 5 个 UDA 方法相对"什么都不做"的提升非常有限，这才是需要向审稿人说明的核心结论；若 source-only 明显更低，则 UDA 确实有效。

---

## 15. 2026-10-21 逐项核对官方实现：8 项偏差修复与修复后全量复查

### 15.0 背景与本轮原则

用户指出：此前在未逐行核对官方实现的情况下声称"代码已与官方方法完全一样"，属于不严谨表述。本轮工作原则：

- **只写有代码证据的结论**：每条"与官方一致"均给出官方文件 + 行号；
- **明确列出全部剩余差异**，不再使用"完全一样"类表述；
- 官方参考来源：DeepCORAL、DANN(fungtion)、CDAN(thuml)、Transfer-Learning-Library(tllib) 均为**仓库内克隆**；PDA 官方仓库（BaiShuanghao/Prompt-based-Distribution-Alignment）无本地克隆，本轮通过抓取官方 `utils/ift_block.py`、`trainers/da/pda.py`、`trainers/baseda.py`、`configs/trainers/PDA/b32_ep20_office31.yaml` 在线核对。

### 15.1 八项偏差与修复（每项附官方行号依据）

| # | 偏差 | 官方依据 | 修复位置 |
|---|------|---------|---------|
| 1 | DeepCORAL 的 CORAL 损失误作用于 512 维池化特征 | 官方 `DeepCORAL/main.py:46` `models.CORAL(out1, out2)`，其中 out1/out2 为 fc 之后 logits（`models.py:40-46`）；CORAL 公式 `models.py:13-28` | `DeepCORAL/train_deepcoral.py:194` 改作用于分类器输出 |
| 2 | MCC/MDD iters_per_epoch 用数据加载器长度（~11-16），训练更新数比官方少约两个数量级 | tllib `examples/.../mcc.py:248`、`mdd.py:249` 默认 `-i 1000` | `MCC/train_mcc.py:147`、`MDD/train_mdd.py:151` 恢复默认 1000 |
| 3 | MCC 源/目标分别前向（含 BatchNorm 的 ResNet 下与官方不等价） | tllib `mcc.py:166-168` `torch.cat((x_s,x_t))` 一次前向 + `y.chunk(2)` | `MCC/train_mcc.py:113-115` 恢复 cat 前向（MDD 原本即正确，`mdd.py:168-171` vs `MDD/train_mdd.py:114-117`） |
| 4 | MDD bottleneck_dim 误为 256 | tllib `mdd.py:227` 默认 1024（且 `mdd.py:75` `width=args.bottleneck_dim`） | `MDD/train_mdd.py:149,191-192` 恢复 1024（MCC 官方默认即 256，`mcc.py:224`，未动） |
| 5 | CDAN 强制使用 RandomLayer | 官方 `CDAN/pytorch/train_image.py:208` `--random type=bool default=False`，`:113-118` random=False 时 `random_layer=None`、ad_net 输入 = `output_num()×class_num`（外积） | `CDAN/train_cdan.py:98-100,146-154` 改为 `--random` 可选、默认 False 走外积分支 |
| 6 | DANN 对抗系数 α 按 epoch 计算且首轮即 p=1/epochs | 官方 `DANN/train/main.py:100-101` 逐迭代 `p=float(i+epoch·len)/n_epoch/len`，`α=2/(1+exp(−10p))−1` | `DANN/train_dann.py:185-186` 恢复逐迭代调度 |
| 7 | PDA 类名文件接线冲突（14 类文件 vs 目标域 9/6/8 类的 expected_count） | 项目规则：标签空间以目标域为准；`tools/class_names.py:7` 严格校验 14 类 | `PDA/train_pda.py:165-179,649` 始终按 14 类加载文件、再按目标域 class_ids 取子集 |
| 8 | 配对数据清单额外丢弃单模态方法可用的 VIS 图像（实测：晴天 178/1136、逆光 15/164，其余 0） | 各对比方法原始设定均为单模态 VIS | `tools/prepare_paired_protocol_lists.py` 新增 `--modality {paired,vis}`；生成 `data_lists_vis/` 并设为 UDA 默认 `--data-dir` |

附带修复（同属配方核对）：DeepCORAL 每轮步数与循环取样（官方 `main.py:29` `train_steps=min(...)`，已对齐 `train_deepcoral.py:174-176`）；DANN 同项（官方 `main.py:93`，已对齐 `train_dann.py:179-181`）；CDAN 训练预算由 epochs×loader 长度改为官方迭代制 `num_iterations=100004`、`test_interval=500`（`train_image.py:217,204`），预处理恢复 `pre_process.py:138-150 image_train(256,224)`，评估恢复官方默认 `test_10crop=True` 十裁剪协议（`train_image.py:22-45,228` + `pre_process.py:169-240`），训练 batch 恢复官方 36（`train_image.py:249-250`）。

### 15.2 修复后逐方法对照结论（一致项均有行号；差异全部列出）

**DeepCORAL**（官方：`DeepCORAL/main.py`、`models.py`、`data_loader.py`，仓库内克隆）

- 一致项：CORAL 公式直接 import 官方 `models.py:13-28`（构造性相同）；CORAL 作用于 fc logits（`main.py:46` vs 我方 `:194`）；每轮 min 步数（`main.py:29` vs `:176`）；λ=(e+1)/E（`main.py:147-148` vs `:171`）；SGD 双参数组 fc 10×lr、momentum 0.9、**不传** weight_decay（`main.py:130-133`，该仓库虽定义 `WEIGHT_DECAY=5e-4` 于 `:14` 但未传给优化器，vs `:147-150`）；fc 初始化 normal_(0,0.005)（`models.py:38` vs `:55`）；损失组合 cls+λ·coral（`main.py:48` vs `:195`）；CE 损失（`main.py:45` vs `:153,192`）；shuffle+drop_last（`data_loader.py:93-94` vs `:137-140`）。
- 显式差异：①主干 AlexNet(227px，`models.py:49-81`)→**ResNet-18(224px)**（项目统一主干，最大适配）；②batch 200/56（`main.py:16`）→32/32；③训练增广：官方仅 Scale+ToTensor+Normalize（`data_loader.py:80-84`），我方**新增了 RandomHorizontalFlip**（本轮新识别，见 15.3）；④评估集：官方 office31 训练域即测试域，我方用目标域 train/val 划分；⑤标签空间以目标域类别为准（项目规则）；⑥EPOCHS 官方 20（`main.py:17`），正式重跑用 20。

**DANN**（官方：`DANN/train/main.py`、`DANN/models/*`，仓库内克隆）

- 一致项：GRL 直接 import 官方 `models/functions.py:4-16`；Adam(lr=1e-3)（`main.py:76` vs 我方 `:159`）；每轮 min 步数（`main.py:93` vs `:181`）；逐迭代 p/α（`main.py:100-101` vs `:185-186`）；损失组合 s_label+s_domain+t_domain（`main.py:148` vs `:207-210`）；域标签 source=0/target=1（`main.py:112,136` vs `:203-204`）；源/目标分别前向（`main.py:125,146` vs `:199,205`）；官方 NLLLoss+LogSoftmax（`main.py:78-79`、`models/model.py:29,36`）与我方 CE(logits) 数学等价。
- 显式差异：①主干 MNIST 小 CNN（`models/model.py:9-18`）→ResNet-18，分类头 100-100-10/域头 100-2（`model.py:20-36`）→256 单隐层+Dropout(0.5)（主干替换的连带适配）；②batch 128（`main.py:21`）→32；③n_epoch 100（`main.py:23`）→正式重跑 20（**训练预算少于官方**）；④图像 28px→224px 且新增 RandomHorizontalFlip（官方 MNIST 流程无增广）；⑤官方每 epoch 存模型并 test，我方不选 best、报告 final+诊断 peak（项目规则）。

**CDAN**（官方：`CDAN/pytorch/*`，仓库内克隆；network/loss/lr_schedule/pre_process 四个模块被我方**直接 import**，构造性逐字节一致）

- 一致项：random=False 分支与 ad_net 输入维度（`train_image.py:113-118` vs 我方 `:146-154`）；迭代制 100004/test_interval 500（`:217,204` vs `:84-87`）；迭代器耗尽重建（`:168-171` vs `:200-203`）；inv 逐迭代 0 基调度（`:166` + `lr_schedule.py:1-10` vs `:197-199`，lr_mult/decay_mult 分组语义由官方 `network.py:230-241,401-402` 提供）；源/目标分别前向后 cat（`:175-178` vs `:214-217`）；softmax(cat)（`:179` vs `:218`）；CDAN+E 默认与 entropy、calc_coeff(i)（`:180-184,198` + `network.py:10-11` vs `:95-97,224-230`）；cls 作用于 outputs_source（`:189` vs `:221`）；trade_off=1.0（`:229` vs `:232`）；SGD(1e-3, mom 0.9, wd 5e-4, nesterov)（`:244-246` vs `:161-162`）；batch 36/36、test batch 4、shuffle+drop_last（`:249-251,86-91` vs `:88,136-139,61`）；image_train(256,224)（`:228` vs `:119`）；十裁剪评估（10 个 transform、softmax 求和 argmax，`:22-45` vs `:54-77`）。
- 显式差异：①主干官方默认 ResNet50（`train_image.py:200`）→ResNet18（**官方支持的可选项**，`:200` choices 含 ResNet18；项目统一主干）；②评估集：官方以目标域 train list 兼作 test（`:251`）→我方目标域 val 划分；③best 选择（`:151-153,193`）→项目禁用，报告 final+诊断 peak；④评估时点：官方在 `i%500==499` 的迭代**开头**评估（即第 499 次更新之前），我方在该迭代**末尾**评估且末次迭代补评一次（差 1 次更新的报告时点，本轮新识别）；⑤num_workers 4→2。

**MCC**（官方：tllib `examples/domain_adaptation/image_classification/mcc.py` + tllib 模块，仓库内克隆；损失/分类器/迭代器/主干均**直接 import tllib**，构造性相同）

- 一致项：bottleneck 256（`mcc.py:224` vs 我方 `:145`）；iters 1000（`:248` vs `:147`）；cat 前向+chunk（`:166-168` vs `:113-115`）；loss=cls+1.0·mcc（`:170-172,230` vs `:116-118,144`）；SGD(get_parameters(), lr 0.005, mom 0.9, wd 1e-3, nesterov)（`:77-78,236-243` vs `:189-190`）；LambdaLR inv γ=0.001/power 0.75 逐迭代 step（`:79,184` vs `:191,121`）；batch 36（`:233` vs `:137`）；temperature 2.5（`:229` vs `:143`）；transforms：ResizeImage(256)+RandomResizedCrop(224, 0.08-1.0, 3/4-4/3)+flip，val ResizeImage(256)+CenterCrop(224)（examples `utils.py:144-216` vs `:64-80`）；epochs 官方 20（`:246`）。
- 显式差异：①best checkpoint（`mcc.py:120-131`）→项目禁用；②epoch 计数 0 基→1 基（仅显示）；③目标域 val 协议；④num_workers 官方 2→我方 0（Windows）。

**MDD**（官方：tllib `mdd.py`，同 MCC 结构）

- 一致项：bottleneck 1024 且 width=bottleneck（`mdd.py:227,75` vs 我方 `:149,191-192`）；margin 4.0（`:231` vs `:147`）；lr 0.004/γ 0.0002/decay 0.75/wd 5e-4/batch 32（`:235-244` vs `:141-146`）；iters 1000（`:249` vs `:151`）；cat 前向+双 chunk（`:168-171` vs `:114-117`）；transfer=−mdd、classifier.step() 先于 backward（`:177-179` vs `:120-122`）；SGD+LambdaLR（`:80-81` vs `:194-196`）；transforms 与 epochs 同 MCC。
- 显式差异：同 MCC 的①-④。

**PDA**（官方：GitHub 在线核对，本轮抓取 `utils/ift_block.py`、`trainers/da/pda.py`、`trainers/baseda.py`、`configs/trainers/PDA/b32_ep20_office31.yaml`）

- 一致项（本轮核对的官方原文）：IFT_Module 的 Q/K/V 拆分为 einops `rearrange(out,'b (d k)->k b d',k=3)`（ift_block.py forward 段），我方 `:350-352` 用 `view(b,-1,3).permute(2,0,1).unbind(0)`——索引语义逐项验证等价（q/k/v 分别为通道 0,3,6,…/1,4,7,…/2,5,8,…）；IFT 其余计算（softmax(scale·q@kᵀ)、Fv+post_project(A@v)、归一化、logit_scale.exp()、logits_s+logits_t）与官方一致；**beta_s/beta_t=0.1**（官方 trainer 在 `pda.py:125` 显式传 0.1，覆盖类默认 1.0，vs 我方 `:436`）；ctx 初始化（官方 `pda.py:46-57` 先取 token 嵌入再立刻 `normal_(0.02)` 覆盖，等效于我方直接 normal 初始化 `:391-393`）；deep prompts 结构（SHARE_LAYER=[0,5]+DEEP_SHARED：proj 克隆 6 个、deep_vctx 为 num_layer−6 层，`pda.py:97-99` vs `:411-415`）；PromptLearner.forward（`pda.py:132-153` vs `:443-451`）；CustomCLIP.forward 全部分支（train 源/目标、construct、eval 的 logits+0.5·logits_c，`pda.py:188-264` vs `:484-558`）；bank 逐样本更新规则（K=5、min-prob 替换、source 侧要求 l==label_p，`pda.py:126,205-228` vs `:437,505-522`）；run_epoch 语义（COUNT_ITER=train_x→num_batches=len(train_x)、迭代器耗尽重建，baseda.py run_epoch 段 vs `:719,725-738`）；冻结策略（仅 prompt_learner 可训练、bank 冻结，`pda.py:279-284` vs `:668-673`）；超参（yaml：batch 32/32/128、SGD lr 0.003 mom 0.9 wd 5e-4、cosine+warmup 1 constant 1e-5、MAX_EPOCH 20、ViT-B/16、CLIP 归一化、random_resized_crop+random_flip，vs `:592-610,147-162,680-683`）。
- 显式差异：①dassl 未移植，训练循环按相同预算语义重写（docstring 早已声明）；②einops→纯 torch 等价写法（已验证）；③仅 ViT-B/16 路径；④torch.jit 加载回退→`build_model(state_dict)`；⑤类名按目标域子集取提示词，无文件时用占位符并记录 prompt_source；⑥best 选择（baseda.py after_epoch 的 best_result）→项目禁用；⑦bank 构造循环加了 `min(probs)>0.99` 早停（工程保护）；⑧官方 `PDA.construct_bank`/`forward_backward` 的 trainer 侧完整源码本轮未逐行核对（在线抓取被截断），已核对的是模型侧 bank 更新规则与 loss 组合。
- 官方 yaml 中 CONFI/EPOCH/WARM_UP 由 `scripts/pda/main_pda.sh` 传入（Office-31 为 0.9/0/1），本轮未重新抓取该 sh，沿用此前移植时的核对结论；我方默认 `--confi 0.9 --warm-up 1 --epoch-switch 0`。

### 15.3 本轮新识别、此前未声明的差异（如实补充）

1. **DeepCORAL 训练增广**：官方 office31 加载器无任何随机增广（`data_loader.py:80-84` 仅 Scale+ToTensor+Normalize），我方训练 transform 含 RandomHorizontalFlip（`train_deepcoral.py:70`）。DANN 同理（官方 MNIST 流程无增广，我方 `:82`）。是否移除增广待用户决策（属于改配方）。
2. **CDAN 评估时点**：官方在 `i%500==499` 迭代开头评估（`train_image.py:146-149`），我方在该迭代末尾评估并末次补评（`train_cdan.py:245-256`）；差 1 次更新的报告时点，不影响训练。
3. **DANN/DeepCORAL 分类头结构**与官方不同（主干替换的连带项，此前只笼统声明"主干适配"，现细化到具体结构，见 15.2）。

### 15.4 数据清单与启动器变更

- 新增 `tools/prepare_paired_protocol_lists.py --modality vis`：VIS-only 清单（同一 union/hash 划分规则，不要求 IR 配对）；paired 模式行为不变，仍供 sda/jmda/oracle 协议使用。
- 已生成 `data_lists_vis/`：sunny_train 884 行；backlight 128/36、rainy 96/26、foggy 440/115、dark 465/120（train/val）。
- `scripts/run_uda_sunny.ps1`：新增 `-DataDir`（默认 `data_lists_vis`）与 `-CdanIterations`（默认 100004）；CDAN 特判传 `--num-iterations`，其余方法传 `--epochs`。
- `scripts/run_server.sh` uda 分支：先生成 VIS 清单到 `${FAIR_LIST_DIR}_vis`，经 `UDA_COMMON` 传 `--data-dir`；CDAN 特判 `${CDAN_NUM_ITERATIONS:-100004}`。
- 配对裁剪的真实影响（实测自 `prepared_lists/local_smoke_20261013` manifest）：晴天 178/1136、逆光 15/164、雨天/雾天/黑天为 0；此前"源域仅剩 515/401/493"主要由目标域类别过滤（用户指定规则）造成，不是配对裁剪。

### 15.5 results_summary.txt 更正与旧结果作废

- 两批冒烟结果的基线标注 "Random baseline (14 classes): 0.0714" **有误**：UDA 标签空间以目标域类别为准，正确基线为 backlight 0.1111（9 类）、rainy 0.1667（6 类）、foggy 0.1111（9 类）、dark 0.1250（8 类），文件顶部已加作废警告。
- `results_summary.txt` 中的 3-epoch 数字与 `efficiency_results/uda_sunny_targetspace/` 的重跑结果均为**修复前配方**产物，已全部作废；正式结果以修复后代码 + `data_lists_vis` 重跑为准。

### 15.6 修复后验证

- 7 个改动文件 `py_compile` 全部通过。
- 5 个非 CLIP 方法最小预算冒烟全部正常完成：DeepCORAL/DANN `--epochs 1`；MCC/MDD `--epochs 1 -i 10`（命名空间确认 iters_per_epoch 可配、MDD bottleneck_dim=1024 生效）；CDAN `--num-iterations 12 --test-interval 6`（十裁剪评估正常，ad_net 输入 2304=256×9 确认 random=False 外积分支生效）。冒烟日志均显示目标域类别过滤工作正常（source kept 584/dropped 300，逆光 9 类）。
- PDA 需下载 CLIP 权重未做运行冒烟，仅编译通过；其 IFT 拆分与超参本轮已对官方源码核对（15.2）。

---

## 16. 2026-10-23 官方公式与独立训练预算修正

本节覆盖第 15 节中后来复核发现不准确的 DeepCORAL 公式判断及统一训练预算。

- **DeepCORAL 公式**：仓库内第三方 PyTorch 端口的 `models.CORAL` 虽然被旧记录
  当作参考实现，但它使用未除 `n-1` 的散布矩阵，并在矩阵均值后再次除
  `4d²`，不符合论文 Eq. (1)--(3)。现已改为无偏协方差
  `centered.T @ centered / (n-1)`，再计算平方 Frobenius **和**并除 `4d²`。
  batch=32 时，旧实现相对论文公式在 6/8/9 类任务上分别放大约
  26.69/15.02/11.86 倍。
- **DeepCORAL 优化器**：恢复论文实验的 `weight_decay=5e-4`，保留新分类层
  10× 学习率及 momentum=0.9；默认训练轮数改为 20。
- **独立官方预算**：DANN/MCC/MDD/PDA 默认轮数分别改为 100/20/20/20；
  CDAN 继续使用 100004 iterations。`run_server.sh` 不再把统一 `EPOCHS=100`
  传给全部 UDA 基线，而是使用可分别覆盖的 `UDA_*_EPOCHS`。
- **Windows 六方法入口**：`run_uda_sunny.ps1` 加入 PDA，要求真实 14 类英文名称
  文件；默认使用上述独立预算。仅当显式传 `-Epochs N` 时，才以统一 N 轮进行
  冒烟测试（CDAN 仍保持迭代制）。
- 目标域类别集合裁剪、源域过滤、连续标签重映射及全量 VIS 清单规则均未改变。

---

## 17. 2026-10-23 CCSA、d-SNE、MCFT 官方训练逻辑修正

经重新逐项对照三个官方仓库，修复此前仅保留核心损失/网络、但训练协议不等价的问题：

- **共同数据口径**：三者均调用 `uda_label_space(data_dir, target)`，按目标天气
  train∪val 的类别支持过滤晴天源域并紧凑重映射标签；分类头宽度改为 `|T|`。
- **CCSA**：恢复全训练池穷举正样本对、3:1负样本对；pair表只在训练前生成并
  打乱；每个pair batch严格执行“源分支CE+CSA更新一次，再反转同一批pair执行
  目标分支CE+CSA更新一次”；Adadelta参数恢复Keras默认语义
  `lr=1, rho=0.95, eps=1e-7`，默认80轮、pair batch=256。
- **d-SNE**：恢复官方默认预激活ResNetV2-101与4096→512 embedding head；恢复穷举正样本
  对及1:1负样本对；默认只执行“目标CE + 目标→源d-SNE”更新，只有显式
  `--train-src`才增加官方可选的源方向更新；默认20轮、batch=32、SGD配置不变。
- **MCFT**：维持官方MViT、ITA、ITM以及`CE+0.1*ITA+ITM`；输入明确映射为
  VIS RGB三通道与IR灰度单通道（对应官方HSI多通道+LiDAR单通道的可用数据适配），
  取消非官方默认水平翻转和ImageNet归一化；默认500轮、每10轮评估、Adam与StepLR
  恢复官方配方。MCFT仍是源域监督多模态分类，不宣称为域适应。
- **评价指标保存**：新增 `tools/metrics_io.py`。每个目标域都会在checkpoint目录写
  `<method>_<target>_seed<seed>_metrics.csv` 与同名JSON，显式保存 ACC、macro Precision、
  macro Recall、macro F1、末次结果、ACC最佳轮及完整评估历史；同时保存last/best模型。
- **启动预算**：`run_server.sh sda` 未显式传轮数时为CCSA/d-SNE分别使用80/20；
  `run_server.sh supervised` 默认MCFT 500轮。第二个位置参数或对应环境变量仍可覆盖。

验证：Python语法编译通过；12项协议/结果单元测试通过；CCSA双更新、d-SNE官方默认
方向更新、MCFT VIS3+IR1 的前向与反向传播均通过；Git Bash `bash -n`检查通过。




