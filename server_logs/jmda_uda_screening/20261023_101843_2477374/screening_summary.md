# JMDA UDA 筛选实验汇总

> 这些结果仅用于定位伪标签、SVD/Tensor 和 OT 的故障来源，不能进入论文的 SDA/UDA 对比表。

| 目标域 | 实验 | 配对来源 | 预热末Acc | 第11轮Acc | 最终Acc | Macro-F1 | 相对预热 | 目标CE=0 |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| backlight | oracle_tensor_only | target_ground_truth_oracle | 36.11% | 52.78% | 11.11% | 7.04% | -25.00% | 是 |
| backlight | oracle_tensor_ot | target_ground_truth_oracle | 30.56% | 50.00% | 75.00% | 74.69% | 44.44% | 是 |
| backlight | pseudo_tensor_only | source_prototype_pseudo_class | 36.11% | 33.33% | 16.67% | 15.38% | -19.44% | 是 |
| backlight | pseudo_tensor_ot | source_prototype_pseudo_class | 38.89% | 30.56% | 2.78% | 2.22% | -36.11% | 是 |

## backlight 诊断

- 实验矩阵不完整，缺少：source_only_fixed_projection。

## 未成功解析的日志

- backlight_source_only_fixed_projection.log: missing screening marker or JMDA_RUN_SUMMARY_JSON
