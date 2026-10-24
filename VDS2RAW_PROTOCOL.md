# VDS2Raw v2 三模态跨日有监督域适应

## 数据与协议

- 本地根目录：`D:\Downloads\VDS2Raw\threeclass_ready`
- 服务器根目录：`/home/lixiang/lx/VDS2Raw/threeclass_ready`
- 类别：Cargo、Fishing、Sailing & Pleasure
- 源域：`source_train`，day3，116 条
- 有标签目标训练域：`target_train`，非 day3，94 条
- 选模集：`target_val`，28 条
- 最终测试集：`target_test`，33 条

训练和统计拟合严格限定为 `source_train + target_train`。每个方法只用
`target_val` 选择最佳 epoch；选定并重新载入最佳检查点后，才构造
`target_test` loader 并执行一次最终评价。最终保存 ACC、宏平均 Precision、
Recall 和 F1。

运行前的结构审计会检查 271 个 NPZ 的键、形状、类型、类别和 manifest
一致性，也会独立验证不同分区的 scene/MMSI 交集为零。该审计不计算训练统计、
不进行模型推理，也不影响选模。RGB/NIR 均值方差仅由 210 条训练样本计算；
AIS 中已经由数据构建程序基于相同训练分区标准化的 `length_z`、`width_z`
直接使用，并与 13 维缺失掩码一同送入模型。

## 方法输入

| 方法 | 输入 | 训练协议 |
|---|---|---|
| CCSA | RGB | 官方正负跨域配对、CSA、双向分类更新 |
| d-SNE | RGB | 官方跨域配对与 d-SNE 距离损失 |
| MCFT-RGB-NIR-AIS | RGB + NIR + AIS + mask | 官方双影像 MCFT 主干，增加显式 AIS/mask 后融合分支 |

MCFT 原论文只有两路栅格模态。因此三模态版本保留其 RGB/NIR Transformer、
ITA、ITM 和优化配方，同时用一个 26 维 `AIS*mask + mask` MLP 分支进行后
融合。这是透明标注的数据集适配版本，不宣称 AIS 分支来自原论文。

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

本地脚本针对 8GB 显存默认使用 CCSA/d-SNE/MCFT 批大小 64/8/32；可以用
`VDS2RAW_CCSA_BATCH_SIZE`、`VDS2RAW_DSNE_BATCH_SIZE` 和
`VDS2RAW_MCFT_BATCH_SIZE` 覆盖。训练轮数默认分别为 80/20/500。

多随机种子示例：

```powershell
$env:VDS2RAW_SEEDS = '42 43 44 45 46'
./scripts/run_vds2raw_methods.ps1 -Mode formal
```

## Linux 服务器运行

```bash
export VDS2RAW_DATA_ROOT=/home/lixiang/lx/VDS2Raw/threeclass_ready
export VDS2RAW_SEEDS="42"
bash scripts/run_vds2raw_background.sh smoke
bash scripts/run_vds2raw_background.sh formal
```

## 输出

每次运行写入 `runs/vds2raw/<运行ID>/`：

- `dataset_audit.json`：分区、NPZ、manifest 和泄漏审计；
- `normalization_stats.json`：只基于两个训练分区的 RGB/NIR 统计；
- `logs/`：各方法标准输出和错误日志；
- `artifacts/*_best.pth`：由 `target_val` 选择的最佳模型；
- `artifacts/*_target_val_metrics.csv/json`：验证曲线；
- `artifacts/*_result.csv/json`：选中验证指标和一次性最终测试指标；
- `summary/runs.csv`：每个方法、每个随机种子的最终四项测试指标；
- `summary/summary.csv`：跨随机种子的均值和总体标准差。

