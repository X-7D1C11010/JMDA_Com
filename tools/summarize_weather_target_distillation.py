"""Audit student-only results of cached target-probability distillation."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_weather_goal import read_run


def main():
    rows=[]
    for directory in sorted((ROOT/'runs/weather_enhanced_screen').glob('prediction_distill_*')):
        for path in directory.rglob('result.json'):
            result=json.loads(path.read_text(encoding='utf-8'));enhancement=result.get('enhancement',{})
            if enhancement.get('variant_family')!='core_plus_cached_ensemble_target_distillation':
                raise ValueError('Unexpected variant: '+str(path))
            row=read_run(path,directory.name)
            row.update(distillation_weight=enhancement['prediction_distillation_weight'],
                cache_start_epoch=enhancement['prediction_distillation_start_epoch'],
                sharpening=enhancement['prediction_distillation_sharpening'],
                encoder_LR_scale_after_cache=enhancement.get('prediction_distillation_encoder_lr_scale',1.),
                inference_teacher_encoders=enhancement['prediction_teacher']['extra_inference_image_encoders'],
                optimizer_plus_teacher_preparation_time_s=result['complexity'].get('optimizer_and_teacher_preparation_time_s'),
                inference_GFLOPs=result['complexity']['inference_GFLOPs'])
            if row['inference_teacher_encoders']!=0:raise ValueError('Inference still includes teacher')
            rows.append(row)
    best=[]
    for weather in ['rainy','foggy','dark','backlight']:
        for method in ['dann','deepcoral','cdan','mdd','mcc']:
            candidates=[r for r in rows if r['target']==weather and r['method']==method]
            if candidates:best.append(max(candidates,key=lambda r:(r['ACC_percent'],r['F1_percent'])))
    rain=[r for r in best if r['target']=='rainy']
    payload={'updated_UTC':datetime.now(timezone.utc).isoformat(),'variant':'original core + cached ensemble target distillation',
        'unmodified_official_method_scores':False,'target_training_ground_truth_used':False,
        'validation_used_for_pseudo_probability_fitting':False,'inference_uses_only_student':True,
        'evaluation':'within-run highest target_val ACC, macro F1 tie-break; no independent weather test',
        'rain_all_five_completed':len(rain)==5,'rain_all_five_above80':len(rain)==5 and all(r['ACC_percent']>=80 for r in rain),
        'all_four_weather_goal_achieved':False,'completed_runs':len(rows),'best_per_method_weather':best,'runs':rows}
    (ROOT/'results/weather_target_distillation_runs.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    lines=['# 缓存目标软概率蒸馏：学生独立推理实测','',
        '保留原五方法核心损失，新增目标KL蒸馏；这是增强版本，不称作未修改的官方方法。',
        '教师仅用源标签和无标签目标训练图像统计；固定一轮源监督后缓存目标软概率。验证不生成训练伪概率。',
        '验证与推理只执行学生；两个额外冻结教师编码器仅在准备训练信号时使用。所有成绩为同一最佳轮次的验证指标。','',
        '|方法|天气|ACC%|Precision宏%|Recall宏%|F1宏%|最佳/完成轮次|适配后峰值ACC%|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in best:
        lines.append(f"|{r['method']}|{r['target']}|{r['ACC_percent']:.2f}|{r['Precision_percent']:.2f}|{r['Recall_percent']:.2f}|"
            f"{r['F1_percent']:.2f}|{r['selected_epoch']}/{r['completed_epochs']}|{r['post_adaptation_peak_ACC_percent']:.2f}|")
    lines+=['',f"共核验{len(rows)}项真实训练；雨天五方法全部达到80%：{payload['rain_all_five_above80']}。",
        'DANN/MDD/MCC的受控学习率测试中，缓存软概率与此前版本逐元素一致；编码器学习率在缓存生成后由1e-5提高至3e-4，分类头学习率保持原值。',
        'CDAN的20轮候选为76.92%，延长40轮候选达到80.77%；Deep CORAL为84.62%。',
        '单个seed通过不表示其他天气或多seed已经通过；雾天、黑天仍未达到80%，原三小时时限未满足。',
        '全部候选、参数、原始路径、训练时间和峰值显存见results/weather_target_distillation_runs.json。']
    (ROOT/'WEATHER_TARGET_DISTILLATION_20261115.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'completed_runs':len(rows),'rain_five_above80':payload['rain_all_five_above80']}))


if __name__=='__main__':main()
