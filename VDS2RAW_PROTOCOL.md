# VDS2Raw 三模态实验协议（MCFT / TIR）

## 数据与选模

- 类别：Cargo、Fishing、Sailing & Pleasure。
- `source_train`：116；`target_train`：94；`target_val`：28；`target_test`：33。
- 为保证对比监督预算一致，MCFT 与 TIR 都使用全部 `source_train + target_train`
  真实标签。TIR 因此是公平对比适配，不再宣称遵循论文的 source-free few-shot
  数据预算；其 TIR、CoOp、LoRA 核心来自官方实现，容量和训练日程使用已经过本项目
  正式实验验证的 full-support 基线。
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
8/10 的 TIR、CoOp prompt 和 LoRA 机制保留。多模态与数据加载部分是项目适配：

- RGB：文件虽使用 uint16 容器，但训练像素99%分位约为179/225/262，超过1000的值
  只是稀疏异常高值。因此按官方8-bit `ToTensor` 语义除以255并截断少量高值，再使用
  CLIP 224×224 归一化；避免按全局最大值缩放导致主体图像整体过暗；
- NIR：训练分区统计 z-score 后，通过初始有效权重0.01的可学习门控残差注入 RGB；
- AIS：只使用与 MCFT 相同的 4 个基础字段，mask 仅作缺失值门控；
- TIR 发布代码中的 `1 - beta * (Sum - 3)` 以5-way的 `Sum=3` 为中点；直接用于三分类
  时所有权重均不小于1，无法抑制 `Sum=K` attention-sink token。三分类因此采用论文
  附录明确提出的通用 K-way 简化排序，三分类权重显式限制在 `[0, 2]`，即保留
  `Sum=K` sink token 的完全抑制和 `Sum=1` 类别特异token的2倍增强；官方五分类
  权重公式保持不变；
- CLIP 权重必须精确匹配全部键和形状，任何缺失、额外或 shape 不匹配都会报错，
  禁止静默漏载。

官方发布代码对每个5-way episode使用完整support set，每轮一次更新，共100轮；没有给出
适用于本项目210样本三模态监督任务的专门日程。当前默认正式配置恢复已经取得更高结果的
100 epoch、batch size 32内存微批、每轮一次full-support optimizer update、固定学习率且
关闭早停。LoRA同时作用于视觉/文本编码器Q/K/V，rank 16 / alpha 8 / dropout 0.25；
LoRA/CoOp/adapter学习率分别为 `2e-4`、`2e-3`、`1e-3`，task weight decay为
`5e-4`，label smoothing为0。源域和目标域监督CE分开计算并以0.5/0.5等权合并；每个域
分别根据自身类别计数使用平方根逆频率权重，避免源域Sailing样本掩盖目标域Sailing不足。
自然无放回shuffle保证每轮全部210个样本各使用一次。NIR门控从0.01开始，AIS门控从1.20开始；
AIS门控使用softplus约束在(0,+inf)，只允许正向贡献，同时兼顾首步
梯度和预训练RGB稳定性；AIS dropout为0.3。默认增强恢复为RGB/NIR同步轻裁剪和水平/垂直
翻转。每轮仍评估target_val和target_test，汇总记录完整100轮中的target_test ACC峰值并
明确test-selection bias。逐轮额外记录源/目标训练ACC和加权CE、TIR sink/discriminative/
clipped token比例、NIR/AIS门控尺度与实际optimizer step数。
VDS2Raw 的 NIR/AIS 分支及固定三分类测试划分不是论文原生设置，必须作为适配披露。
所有官方协议差异均写入结果元数据。

若本地还没有官方 CLIP 权重，可在可联网机器执行以下命令。下载器同时校验官方
SHA-256 和完整结构，随后把 `pretrained_weights/clip/ViT-B-16.pt` 上传到服务器：

```bash
python tools/download_tir_clip.py
```

## 启动

服务器正式运行（当前默认只运行仍需优化的 TIR；如需复跑 MCFT 可显式设置方法顺序）：

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

TIR smoke 使用batch 32遍历完整210样本并累积为一次full-support optimizer update，不再
使用只含2个样本的截断batch；这样可验证两域分离损失、域内类别权重与门控的完整路径。

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
