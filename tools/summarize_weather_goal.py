"""Audit completed real method runs against the full 5-method/4-weather goal."""
import csv
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

METHODS=("dann","deepcoral","cdan","mdd","mcc")
TARGETS=("backlight","rainy","foggy","dark")
GOAL_CREATED_UTC_EPOCH=1794632065
GOAL_DEADLINE_UTC_EPOCH=GOAL_CREATED_UTC_EPOCH+3*60*60


def read_run(path,variant):
    r=json.loads(path.read_text(encoding="utf-8"))
    c=json.loads((path.parent/"completeness.json").read_text(encoding="utf-8"))
    config=json.loads((path.parent/"run_config.json").read_text(encoding="utf-8"))
    if c["status"]!="PASS" or r["target_training_labels_used"] is not False or r.get("smoke"):
        raise ValueError("Result is not a complete real UDA run: "+str(path))
    ev=[json.loads(line) for line in (path.parent/"evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    vals=[x for x in ev if x["partition"]=="target_val"]
    selected=[x for x in vals if x["epoch"]==r["selected_epoch"]]
    if len(selected)!=1 or abs(selected[0]["metrics"]["accuracy"]-r["target_val"]["accuracy"])>1e-10:
        raise ValueError("Selected result disagrees with epoch log")
    if abs(max(x["metrics"]["accuracy"] for x in vals)-r["target_val"]["accuracy"])>1e-10:
        raise ValueError("Result is not the within-run peak")
    confusion=selected[0]["metrics"]["confusion_matrix"]
    total=sum(map(sum,confusion));correct=sum(confusion[i][i] for i in range(len(confusion)))
    if abs(correct/total-r["target_val"]["accuracy"])>1e-10:raise ValueError("Confusion/ACC disagreement")
    warmup=config["arguments"]["adaptation_warmup_epochs"]
    post=[x["metrics"]["accuracy"] for x in vals if x["epoch"]>warmup]
    return {"variant":variant,"method":r["method"],"target":r["target"],"seed":r["seed"],
        "ACC_percent":100*r["target_val"]["accuracy"],"Precision_percent":100*r["target_val"]["precision_macro"],
        "Recall_percent":100*r["target_val"]["recall_macro"],"F1_percent":100*r["target_val"]["f1_macro"],
        "selected_epoch":r["selected_epoch"],"completed_epochs":r["completed_epochs"],
        "unblended_student_ACC_percent":(100*selected[0]["unblended_student_metrics"]["accuracy"]
            if "unblended_student_metrics" in selected[0] else None),
        "planned_epochs":r["planned_epochs"],"selected_before_adaptation":r["selected_epoch"]<=warmup,
        "core_trade_off":config["arguments"]["trade_off"],
        "adaptation_warmup_epochs":warmup,"adaptation_ramp_epochs":config["arguments"]["adaptation_ramp_epochs"],
        "post_adaptation_peak_ACC_percent":100*max(post) if post else None,
        "image_backbone":r["image_backbone"],"IR_backbone":r.get("ir_backbone",r["image_backbone"]),
        "encoder_scope":r.get("encoder_training_scope",config.get("encoder_training_scope",
            "frozen" if r.get("freeze_image_encoders") else "all")),"precision":r.get("encoder_precision","fp32"),
        "source_batch_size":config["arguments"]["batch_size"],
        "target_batch_size":config["arguments"].get("target_batch_size") or config["arguments"]["batch_size"],
        "adversarial_lr_override":config["arguments"].get("adversarial_lr_override"),
        "mcc_temperature":config["arguments"].get("mcc_temperature",2.5) if r["method"]=="mcc" else None,
        "VIS_gradient_checkpointing":config["arguments"].get("checkpoint_vis_blocks",False),
        "training_time_s":r["complexity"]["total_train_time_s"],
        "peak_reserved_GiB":r["complexity"]["peak_train_reserved_gb"],
        "source_weather_probability":config["arguments"].get("source_weather_probability",0.),
        "source_aspect_min":config["arguments"].get("source_aspect_min",1.),
        "source_aspect_max":config["arguments"].get("source_aspect_max",1.),
        "result_completed_UTC_epoch":path.stat().st_mtime,
        "result_file":str(path.relative_to(ROOT)).replace('\\','/')}


def main():
    rows=[];base=ROOT/"runs/weather_local_screen"
    for directory in sorted(base.glob("goal3h_*")):
        if (directory/"aborted.json").exists():continue
        for path in sorted(directory.glob("*/*_seed42/result.json")):
            rows.append(read_run(path,directory.name))
    earlier=[]
    with (ROOT/"results/weather_backlight_seed42_qualified.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):earlier.append(read_run(ROOT/r["result_file"],"earlier_qualified_backlight"))
    matrix=[]
    for target in TARGETS:
        for method in METHODS:
            candidates=[r for r in rows+earlier if r["method"]==method and r["target"]==target]
            best=max(candidates,key=lambda r:(r["ACC_percent"],r["F1_percent"])) if candidates else None
            matrix.append({"target":target,"method":method,"best":best,
                "qualified":bool(best and best["ACC_percent"]>=80.-1e-10),
                "qualified_within_original_window":any(r["ACC_percent"]>=80.-1e-10 and
                    (r["variant"]=="earlier_qualified_backlight" or r["result_completed_UTC_epoch"]<=GOAL_DEADLINE_UTC_EPOCH)
                    for r in candidates)})
    stamp=datetime.now(timezone.utc)
    payload={"updated_UTC":stamp.isoformat(),"updated_Beijing":stamp.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "objective":"all 5 methods on all 4 weathers >=80% real single-seed peak ACC",
        "completed_new_method_runs":len(rows),"goal_qualified_pairs":sum(x["qualified"] for x in matrix),
        "goal_required_pairs":20,"performance_goal_achieved":all(x["qualified"] for x in matrix),
        "goal_achieved":all(x["qualified_within_original_window"] for x in matrix),
        "original_3h_deadline_UTC":datetime.fromtimestamp(GOAL_DEADLINE_UTC_EPOCH,timezone.utc).isoformat(),
        "original_3h_window_elapsed":stamp.timestamp()>GOAL_DEADLINE_UTC_EPOCH,
        "diagnostic_classifier_scores_included":False,"trained_model_files_saved":False,
        "evaluation_partition":"target_val; no separate weather test; peak within each run",
        "runs":rows,"coverage":matrix}
    (ROOT/"results/goal3h_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    if rows:
        with (ROOT/"results/goal3h_runs.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# 天气实验三小时优化记录","",f"更新：{payload['updated_Beijing']}","",
        f"已完成新候选{len(rows)}项；达到80%的方法/天气组合{payload['goal_qualified_pairs']}/20。",
        "原三小时窗口已结束，时间与准确率要求未同时满足。" if payload["original_3h_window_elapsed"] and not payload["goal_achieved"] else "原三小时窗口尚未结束。",
        "逆光沿用此前真实seed42的20/60轮前缀证据，不能冒称新完整实验。其余单元格为本窗口已完成候选的最高值；未测项目不会以诊断分数填充。","",
        "| 方法 | 逆光 | 雨天 | 雾天 | 黑天 |","|---|---:|---:|---:|---:|"]
    for method in METHODS:
        cells=[]
        for target in TARGETS:
            item=next(x for x in matrix if x["method"]==method and x["target"]==target)
            cells.append(f"{item['best']['ACC_percent']:.2f}%" if item["best"] else "未完成本轮候选")
        lines.append("| "+method+" | "+" | ".join(cells)+" |")
    lines += ["","各候选完整ACC/Precision/Recall/F1、最佳轮次、适配后峰值、配置及原始路径见results/goal3h_runs.csv及results/goal3h_results.json。",
        "冻结特征、文本、图传播等诊断未列为五方法正式成绩。失败/中断队列保留原始记录并排除。",
        "目标尚未达成。" if not payload["goal_achieved"] else "全部单seed峰值达到阈值；完整预算/多seed稳定性需另行验证。"]
    (ROOT/"WEATHER_OPTIMIZATION_3H_20261114.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:payload[k] for k in ("completed_new_method_runs","goal_qualified_pairs","goal_required_pairs","goal_achieved")}))


if __name__=="__main__":main()
