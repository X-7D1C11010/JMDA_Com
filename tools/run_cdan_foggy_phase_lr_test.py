"""One fixed-seed CDAN test: change one encoder or discriminator LR setting."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_label_guided_matrix import build_jobs
from tools.run_cdan_foggy_alignment_sweep import read,write,comparison_row

DEFAULT_BASELINE=(ROOT/'runs/weather_label_guided/cdan_foggy_alignment0p2_seeds43_46_20261118_103331_480'
                  /'weight0p2/cdan_foggy/seed46/cdan_foggy_seed46/result.json')


def remaining_seed_gate(accuracy,threshold,pilot_seed):
    if not 0<=accuracy<=100 or not 0<=threshold<=100 or pilot_seed not in range(42,47):
        raise ValueError('Invalid real validation score, threshold or pilot seed')
    return [s for s in range(42,47) if s!=pilot_seed] if accuracy>=threshold else []


def build_test_job(original,output,factor=.3):
    old=original['values']
    if (old['method']!='cdan' or old['target']!='foggy' or old['epochs']!=40 or
            old['target_labels_per_class']!=3 or old['matched_alignment_weight']!=.2 or
            old['image_adaptation_lr_scale']!=.1):
        raise ValueError('Expected the completed .2/3-shot/40-epoch/phase-LR .1 baseline')
    if not 0<factor<=1 or factor==.1:raise ValueError('Choose a different positive phase-LR factor, at most 1')
    job=build_jobs(Path(output),Path(old['data_dir']),['cdan_foggy'],[old['seed']],ROOT,True,
        overrides={'matched_alignment_weight':.2,'image_adaptation_lr_scale':factor},
        evaluation_mode='validation_ce',unified_fewshot=True,fewshot_alignment_variant=True)[0]
    changed={k for k in set(old)|set(job['values']) if old.get(k)!=job['values'].get(k)}
    if changed!={'image_adaptation_lr_scale','configuration_tag','save_dir'}:
        raise ValueError(f'Unexpected training changes: {sorted(changed)}')
    job['changed_argument_keys']=sorted(changed)
    return job


def build_discriminator_job(original,output,learning_rate=3e-4):
    old=original['values']
    if (old['method']!='cdan' or old['target']!='foggy' or old['epochs']!=40 or
            old['target_labels_per_class']!=3 or old['matched_alignment_weight']!=.2 or
            old['image_adaptation_lr_scale']!=.1 or old.get('adversarial_lr_override') is not None):
        raise ValueError('Expected the original .2/3-shot/40-epoch baseline with no discriminator override')
    if not 0<learning_rate<float('inf'):raise ValueError('Discriminator LR must be finite and positive')
    job=build_jobs(Path(output),Path(old['data_dir']),['cdan_foggy'],[old['seed']],ROOT,True,
        overrides={'matched_alignment_weight':.2,'adversarial_lr_override':learning_rate},
        evaluation_mode='validation_ce',unified_fewshot=True,fewshot_alignment_variant=True)[0]
    changed={k for k in set(old)|set(job['values']) if old.get(k)!=job['values'].get(k)}
    if changed!={'adversarial_lr_override','configuration_tag','save_dir'}:
        raise ValueError(f'Unexpected training changes: {sorted(changed)}')
    job['changed_argument_keys']=sorted(changed)
    return job


def build_source_ema_job(original,output,momentum=.9,domain_confusion_weight=0.,classifier_layout=None):
    old=original['values']
    if (old['method']!='cdan' or old['target']!='foggy' or old['epochs']!=40 or
            old['target_labels_per_class']!=3 or old['matched_alignment_weight']!=.2 or
            old['image_adaptation_lr_scale']!=.1 or old.get('adversarial_lr_override') is not None):
        raise ValueError('Expected the original .2/3-shot baseline without the failed LR variants')
    if not 0<=momentum<1:raise ValueError('Source EMA momentum must be finite and in [0,1)')
    if not 0<=domain_confusion_weight<float('inf'):raise ValueError('Invalid extra domain-confusion weight')
    overrides={'matched_alignment_weight':.2,'source_anchor_ema_momentum':momentum}
    if domain_confusion_weight:overrides['domain_confusion_weight']=domain_confusion_weight
    if classifier_layout is not None:
        if classifier_layout not in {'linear','reference'}:raise ValueError('Unknown method classifier layout')
        if classifier_layout!=old['classifier_layout']:overrides['classifier_layout']=classifier_layout
    job=build_jobs(Path(output),Path(old['data_dir']),['cdan_foggy'],[old['seed']],ROOT,True,
        overrides=overrides,
        evaluation_mode='validation_ce',unified_fewshot=True,fewshot_alignment_variant=True)[0]
    changed={k for k in set(old)|set(job['values']) if old.get(k)!=job['values'].get(k)}
    expected={'source_anchor_ema_momentum','configuration_tag','save_dir'}
    if domain_confusion_weight:expected.add('domain_confusion_weight')
    if 'classifier_layout' in overrides:expected.add('classifier_layout')
    if changed!=expected:
        raise ValueError(f'Unexpected training changes: {sorted(changed)}')
    job['changed_argument_keys']=sorted(changed)
    return job


def save_report(output,rows,test='encoder_phase_lr'):
    write(output/'comparison.json',rows)
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    discriminator=test=='discriminator_lr'
    ema=test=='source_anchor_ema'
    key='source_anchor_policy' if ema else 'discriminator_initial_lr' if discriminator else 'encoder_phase_lr_factor'
    label='源类别中心' if ema else '判别器初始学习率' if discriminator else '适配阶段骨干LR倍率'
    title=('# 雾天CDAN源EMA与无标签域混淆增强测试' if ema and rows[-1].get('domain_confusion_weight',0)>0
           else '# 雾天CDAN'+label+'单因素测试')
    lines=[title,'',
        '每类3个目标标签；额外类别对齐权重0.2；每次40轮；固定原seed及标签池。'
        '最高验证ACC及同轮宏平均指标，单位%。固定留出集不评价。','',
        '|'+label+'|Seed|峰值轮次|ACC|Precision|Recall|F1|末轮ACC|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row[key]}|{row['seed']}|{row['peak_epoch']}|"+
                     '|'.join(f'{row[k]:.2f}' for k in ('ACC','Precision','Recall','F1','last_ACC'))+'|')
    if len(rows)==2:
        lines+=['',f"相对基线最高ACC变化：{rows[1]['ACC']-rows[0]['ACC']:+.2f}个百分点。单seed结果不计算跨seed标准差。"]
    lines+=['',('分类头/判别器/骨干学习率与骨干适配倍率0.1' if ema else
                '分类头/骨干学习率与骨干适配倍率0.1' if discriminator else '分类头/判别器学习率')+
            '、验证损失平台调度、原生CDAN损失及系数均保持不变。'
            '新增每轮采样原生CDAN与额外类别对齐的独立梯度日志，不改变训练损失。','']
    if ema and rows[-1].get('domain_confusion_weight',0)>0:
        lines+=['额外声明的增强：无标签域混淆BCE，权重'+str(rows[-1]['domain_confusion_weight'])+
                '；域判别器参数及类别概率均detach，仅向特征分支传播额外梯度，不增加目标真标签。','']
    if ema and rows[-1].get('classifier_layout')=='reference':
        lines+=['方法网络结构变体：启用当前代码中CDAN的256维可训练线性瓶颈；原linear布局使用Identity瓶颈。'
                'VIS/IR骨干与特征拼接保持原设置，结构变更单独记录。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-result',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--factor',type=float,default=.3)
    parser.add_argument('--test',choices=('encoder_phase_lr','discriminator_lr','source_anchor_ema'),default='encoder_phase_lr')
    parser.add_argument('--discriminator-lr',type=float,default=3e-4)
    parser.add_argument('--ema-momentum',type=float,default=.9)
    parser.add_argument('--domain-confusion-weight',type=float,default=0.)
    parser.add_argument('--classifier-layout',choices=('linear','reference'))
    parser.add_argument('--run-remaining-after-threshold',action='store_true')
    parser.add_argument('--threshold-acc',type=float,default=65.)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.domain_confusion_weight and args.test!='source_anchor_ema':
        parser.error('Declare the extra domain-confusion loss only in the source-EMA enhancement test')
    if args.classifier_layout is not None and args.test!='source_anchor_ema':
        parser.error('Declare a classifier-layout change in the source-EMA enhancement test')
    if not 0<=args.threshold_acc<=100:parser.error('Threshold ACC must be finite and in [0,100]')
    baseline=args.baseline_result.resolve();result=read(baseline);seed=result['seed']
    if args.run_remaining_after_threshold and seed not in range(42,47):
        parser.error('The pilot must be one of seeds42-46 for the remaining-four rule')
    parent_plan=read(baseline.parents[3]/'plan.json')
    original=next(j for j in parent_plan['jobs'] if j['name']==f'cdan_foggy_seed{seed}')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix=('cdan_foggy_source_ema' if args.test=='source_anchor_ema' else
            'cdan_foggy_discriminator_lr' if args.test=='discriminator_lr' else 'cdan_foggy_phase_lr')
    output=(args.output or ROOT/'runs/weather_label_guided'/f'{prefix}_seed{seed}_{stamp}').resolve()
    if not output.is_relative_to(ROOT/'runs/weather_label_guided'):parser.error('Use runs/weather_label_guided')
    if output.exists():raise FileExistsError(output)
    discriminator=args.test=='discriminator_lr'
    ema=args.test=='source_anchor_ema'
    parameter='source_anchor_ema_momentum' if ema else 'adversarial_lr_override' if discriminator else 'image_adaptation_lr_scale'
    row_key='source_anchor_policy' if ema else 'discriminator_initial_lr' if discriminator else 'encoder_phase_lr_factor'
    old_value=(next(g['lr'] for g in read(baseline.parent/'run_config.json')['optimizer_groups']
                    if g['role']=='new_layers') if discriminator else 'current_batch' if ema else .1)
    new_value=args.ema_momentum if ema else args.discriminator_lr if discriminator else args.factor
    job=(build_source_ema_job(original,output/'trial',new_value,args.domain_confusion_weight,args.classifier_layout) if ema else
         build_discriminator_job(original,output/'trial',new_value) if discriminator else
         build_test_job(original,output/'trial',new_value))
    baseline_sampling=result['label_guided_sampling']
    rows=[{**comparison_row(baseline,baseline_sampling),row_key:old_value}]
    if ema:
        rows[0]['domain_confusion_weight']=0.
        rows[0]['classifier_layout']=original['values']['classifier_layout']
    if args.execute:
        from tools.run_weather_fewshot_suite import check_environment
        check_environment({'jobs':[job]},Path(os.environ.get('JMDA_DATA_ROOT',r'D:\Code\JMDA-Net\Data')))
        from weather.train_vds_core import prepare_datasets
        from weather.fewshot import TargetLabelBudget
        datasets,audit=prepare_datasets(Path(job['values']['data_dir']),'foggy',True)
        budget=TargetLabelBudget.from_dataset(datasets[1],audit['compact_label_map'],3,seed)
        if sorted(budget.target_lookup.items())!=sorted((x['path'],x['label']) for x in baseline_sampling['selected_labelled_samples']):
            raise ValueError('The original fixed target label pool changed')
    output.mkdir(parents=True)
    write(output/'plan.json',{'baseline_result':str(baseline),'seed':seed,'epochs':40,
        'test':args.test,'only_training_parameter_changed':parameter if not args.domain_confusion_weight else None,
        'declared_training_changes':{k:job['values'][k] for k in job['changed_argument_keys'] if k not in {'save_dir','configuration_tag'}},
        'old_value':old_value,'new_value':new_value,
        'run_remaining_after_threshold':args.run_remaining_after_threshold,'threshold_ACC':args.threshold_acc,
        'heldout_evaluation':False,'job':job})
    save_report(output,rows,args.test)
    print('PHASE_LR_TEST_DIRECTORY='+str(output),flush=True)
    if not args.execute:
        write(output/'status.json',{'status':'PLAN_ONLY','completed':0,'total':1});return
    write(output/'status.json',{'status':'RUNNING','completed':0,'total':1,'seed':seed})
    command=[sys.executable,'-B','-u','-X','utf8',str(ROOT/'tools/run_label_guided_matrix.py'),
        '--unified-fewshot','--fewshot-alignment-variant','--pairs','cdan_foggy','--seeds',str(seed),
        '--data-dir',job['values']['data_dir'],'--matched-alignment-weight','0.2',
        '--target-labels-per-class','3',
        '--source-anchor-ema-momentum' if ema else '--adversarial-lr-override' if discriminator else '--image-adaptation-lr-scale',str(new_value),
        '--complete-budget','--early-stopping-patience','0','--evaluation-mode','validation_ce',
        '--local','--local-lease-timeout-seconds','0','--output',str(output/'trial'),'--execute']
    if args.domain_confusion_weight:command+=['--domain-confusion-weight',str(args.domain_confusion_weight)]
    if args.classifier_layout is not None and args.classifier_layout!=original['values']['classifier_layout']:
        command+=['--classifier-layout',args.classifier_layout]
    env=dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1',JMDA_DETERMINISTIC='1',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_HOME=str(ROOT/'pretrained_weights/torch'))
    completed_runs=0
    try:
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        path=Path(job['result_file'])
        row={**comparison_row(path,baseline_sampling),row_key:f'source_ema_m{new_value}' if ema else new_value}
        if ema:
            row['domain_confusion_weight']=args.domain_confusion_weight
            row['classifier_layout']=job['values']['classifier_layout']
        old_args=read(baseline.parent/'run_config.json')['arguments']
        new_args=read(path.parent/'run_config.json')['arguments']
        changed={k for k in set(old_args)|set(new_args) if old_args.get(k)!=new_args.get(k)}
        expected_changes={parameter,'configuration_tag','save_dir'}
        if args.domain_confusion_weight:expected_changes.add('domain_confusion_weight')
        if args.classifier_layout is not None and args.classifier_layout!=old_args['classifier_layout']:
            expected_changes.add('classifier_layout')
        if changed!=expected_changes:
            raise ValueError(f'Actual configuration differs in unexpected settings: {sorted(changed)}')
        if discriminator:
            groups=read(path.parent/'run_config.json')['optimizer_groups']
            classifier_lr=next(g['lr'] for g in groups if g['role']=='new_layers')
            discriminator_lr=next(g['lr'] for g in groups if g['role']=='adversarial')
            if classifier_lr!=old_value or discriminator_lr!=new_value:
                raise ValueError('Classifier/discriminator learning rates do not match the controlled test')
        events=[json.loads(s) for s in (path.parent/'gradients.jsonl').read_text().splitlines()]
        if not all({'core','matched_class_alignment'}<=e['enhancement_components'].keys() for e in events):
            raise ValueError('Independent native/alignment gradient records are missing')
        rows.append(row);save_report(output,rows,args.test)
        completed_runs=1
        write(output/'status.json',{'status':'PASS','completed':1,'total':1,'seed':seed})
        print('PHASE_LR_RESULT='+json.dumps(row),flush=True)
        if args.run_remaining_after_threshold:
            remaining=remaining_seed_gate(row['ACC'],args.threshold_acc,seed)
            if not remaining:
                write(output/'status.json',{'status':'BELOW_THRESHOLD','completed':1,'total':5,
                    'pilot_seed':seed,'pilot_ACC':row['ACC'],'threshold_ACC':args.threshold_acc,'remaining_started':False})
                return
            write(output/'locked_training_configuration.json',{'pilot_result':str(path),'pilot_ACC':row['ACC'],
                'threshold_ACC':args.threshold_acc,'remaining_seeds':remaining,'training_arguments':job['values']})
            write(output/'status.json',{'status':'RUNNING_REMAINING','completed':1,'total':5,
                'pilot_seed':seed,'pilot_ACC':row['ACC'],'remaining_seeds':remaining})
            follow=command.copy()
            start=follow.index('--seeds')+1
            follow[start:start+1]=[str(s) for s in remaining]
            follow[follow.index('--output')+1]=str(output/'remaining')
            subprocess.run(follow,cwd=ROOT,env=env,check=True)
            from weather.summarize_class_matching import summarize
            from tools.run_label_guided_matrix import write_peak_summary,write_pair_summary
            result_paths=[path]+[output/f'remaining/cdan_foggy/seed{s}/cdan_foggy_seed{s}/result.json' for s in remaining]
            all_rows=summarize(result_paths,output/'five_seed_summary')
            for result_path in result_paths[1:]:
                follow_args=read(result_path.parent/'run_config.json')['arguments']
                differences={k for k in set(new_args)|set(follow_args) if new_args.get(k)!=follow_args.get(k)}
                if differences!={'seed','save_dir'}:
                    raise ValueError('The locked configuration changed across seeds: '+str(sorted(differences)))
            write_pair_summary(output/'five_seed_summary',all_rows,list(range(42,47)))
            write_peak_summary(output/'five_seed_summary',all_rows,list(range(42,47)))
            completed_runs=5
            write(output/'status.json',{'status':'PASS','completed':5,'total':5,'pilot_seed':seed,
                'pilot_ACC':row['ACC'],'threshold_ACC':args.threshold_acc,'all_seeds':[42,43,44,45,46]})
            print('FIVE_SEED_SUMMARY='+str(output/'five_seed_summary/peak_acc_summary.csv'),flush=True)
    except BaseException as error:
        write(output/'status.json',{'status':'FAIL','completed':completed_runs,
            'total':5 if args.run_remaining_after_threshold else 1,'error':str(error)})
        raise


if __name__=='__main__':main()
