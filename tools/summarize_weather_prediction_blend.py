"""Report actual completed prediction-blend runs separately from original cores."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_weather_goal import read_run


def main():
    rows=[]
    for directory in sorted((ROOT/'runs/weather_enhanced_screen').glob('prediction_blend_*_seed42_v1')):
        for path in directory.rglob('result.json'):
            result=json.loads(path.read_text(encoding='utf-8'))
            if result.get('enhancement',{}).get('variant_family')!='core_plus_frozen_cluster_probability_blend':
                raise ValueError('Unexpected variant: '+str(path))
            row=read_run(path,directory.name)
            teacher=result['enhancement']['prediction_teacher']
            row.update(teacher_probability_weight=teacher['probability_weight'],
                teacher_temperature=teacher['teacher_temperature'],student_temperature=teacher['student_temperature'],
                extra_frozen_inference_encoders=teacher['extra_inference_image_encoders'])
            rows.append(row)
    rain=[x for x in rows if x['target']=='rainy']
    required={'dann','deepcoral','cdan','mdd','mcc'}
    complete={x['method'] for x in rain}==required
    payload={'updated_UTC':datetime.now(timezone.utc).isoformat(),'variant':'additional frozen-cluster probability ensemble',
        'original_core_scores':False,'target_training_labels_used':False,'validation_used_for_teacher_fit':False,
        'validation_used_for_hyperparameters_and_epoch_selection':True,'rain_all_five_completed':complete,
        'rain_all_five_above_80':complete and all(x['ACC_percent']>=80 for x in rain),
        'all_four_weather_goal_achieved':False,'scores_from_other_diagnostics_counted':False,'runs':rows}
    (ROOT/'results/weather_prediction_blend_runs.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    lines=['# 固定聚类教师概率集成：真实训练记录','',
        '这是原核心加额外推理教师的增强版本，不混称官方原方法成绩。',
        '教师仅用源训练标签及无标签目标训练图像拟合；验证用于选温度、权重及最佳轮次，不参与教师拟合或梯度更新。',
        '每种方法增加两个冻结的ImageNet ResNet-18教师编码器，仍输入原VIS/IR配对；额外参数与推理计算纳入原复杂度统计。',
        '以下为真实完整预算内同一最佳轮次的验证指标；没有独立天气测试集。','',
        '|方法|目标域|ACC%|Precision宏%|Recall宏%|F1宏%|最佳/完成轮次|同轮学生独立ACC%|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['method']}|{r['target']}|{r['ACC_percent']:.2f}|{r['Precision_percent']:.2f}|"
            f"{r['Recall_percent']:.2f}|{r['F1_percent']:.2f}|{r['selected_epoch']}/{r['completed_epochs']}|"
            f"{r['unblended_student_ACC_percent']:.2f}|")
    lines+=['',f"已完成{len(rows)}项。雨天五方法全部完成：{complete}；全部超过80%：{payload['rain_all_five_above_80']}。",
        '这些峰值若出现在适配warmup之前，不能归因为原核心域对齐的提升。学生独立指标和适配后的峰值保留在JSON与逐轮日志中。',
        '雾天、黑天仍未达到80%；此表不能代表所有天气达标。原三小时时限未满足。',
        '详细超参数、原始结果路径、训练时间和峰值显存见results/weather_prediction_blend_runs.json。']
    (ROOT/'WEATHER_PROBABILITY_BLEND_20261115.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'completed':len(rows),'rain_five_completed':complete,'rain_five_above80':payload['rain_all_five_above_80']}))


if __name__=='__main__':main()
