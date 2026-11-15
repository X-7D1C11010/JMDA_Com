"""Rebuild controlled follow-up comparisons from verified real result files."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_weather_goal import read_run


def main():
    prefix="runs/weather_local_screen/"
    strong=prefix+"goal3h_stronger_adaptation_seed42/"
    temp=prefix+"goal3h_mcc_temperature_seed42/"
    full=prefix+"goal3h_fullVIS_capacity_seed42/"
    mcc_strong=strong+"mcc_foggy_stronger_DA/mcc_foggy_seed42/result.json"
    mcc_t1=temp+"mcc_foggy_temperature1/mcc_foggy_seed42/result.json"
    cases=[
        ("DANN DA权重",prefix+"goal3h_fullIR_slowAdversary_seed42/dann_foggy_full_tiny_weather_aspect/dann_foggy_seed42/result.json",
         strong+"dann_foggy_stronger_DA/dann_foggy_seed42/result.json","DA weight .05 ->1.0"),
        ("MCC DA权重",prefix+"goal3h_mcc_targetBatch32_seed42/mcc_foggy_targetBatch32/mcc_foggy_seed42/result.json",mcc_strong,"DA weight .05 ->1.0"),
        ("MCC 温度0.5",mcc_strong,temp+"mcc_foggy_temperature05/mcc_foggy_seed42/result.json","MCC temperature2.5 ->.5"),
        ("MCC 温度1.0",mcc_strong,mcc_t1,"MCC temperature2.5 ->1.0"),
        ("MCC 全VIS微调",mcc_t1,full+"mcc_foggy_fullVIS/mcc_foggy_seed42/result.json","VIS last stage ->all stages; memory-saving recomputation"),
        ("MCC VIS初值0.9",mcc_t1,full+"mcc_foggy_VISprior90/mcc_foggy_seed42/result.json","initial VIS gate .5 ->.9"),
    ]
    rows=[]
    for name,old,new,factor in cases:
        a,b=read_run(ROOT/old,"reference"),read_run(ROOT/new,name)
        rows.append({"name":name,"changed_factor":factor,"baseline":a,"candidate":b,
                     "ACC_change_points":b["ACC_percent"]-a["ACC_percent"]})
    data={"updated_UTC":datetime.now(timezone.utc).isoformat(),"target":"foggy","seed":42,
          "evaluation":"within-run target_val peak; Precision/Recall/F1 from same selected epoch",
          "independent_seed_replicates":False,"target_training_labels_used":False,"achieved_80_percent":False,
          "comparisons":rows}
    out=ROOT/"results/weather_loss_temperature_comparison_20261114"
    out.with_suffix(".json").write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# 雾天：按训练信号进行的真实单因素测试","",
        "seed42，VIS+IR，目标训练真实标签不进入损失；全部为同一验证集上的候选筛选，非独立多seed重复。下表ACC取各自最高轮次。","",
        "| 改动 | 基线ACC | 候选ACC | 变化/百分点 | 候选F1宏 | 最佳/实际epoch |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        a,b=row["baseline"],row["candidate"]
        lines.append(f"| {row['name']} | {a['ACC_percent']:.2f}% | {b['ACC_percent']:.2f}% | {row['ACC_change_points']:+.2f} | {b['F1_percent']:.2f}% | {b['selected_epoch']}/{b['completed_epochs']} |")
    lines += ["","MCC目标批次增大后，适配权重与温度调整继续提高雾天ACC至52.17%，仍比80%低27.83个百分点。",
        "DANN增大适配权重下降；MCC开放VIS全部阶段、或把VIS权重初值改为0.9也下降。这些下降候选没有替换已选配置。",
        "全VIS真实CUDA预检通过，首末卷积均实际更新，预检峰值reserved2.672GiB；正式训练峰值3.068GiB。预检仅1batch且smoke=True，未作为准确率结果。",
        "激活重计算保留原始ConvNeXt结构，不保存训练模型。相关前向/梯度/RNG检查和真实CLI检查通过。",
        "三小时目标未达成；完整四天气五方法概览见根目录WEATHER_OPTIMIZATION_3H_20261114.md。",
        "额外统计校准/无标签聚类初始化尚待用户决定是否作为单独增强变体；没有接入这些训练。",
        "所有候选的ACC、P/R/F、参数、原始日志路径及复杂性保留在同名JSON及全局goal3h_runs.csv。"]
    out.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps([{"name":x["name"],"ACC_percent":x["candidate"]["ACC_percent"],"delta":x["ACC_change_points"]} for x in rows]))


if __name__=="__main__":main()
