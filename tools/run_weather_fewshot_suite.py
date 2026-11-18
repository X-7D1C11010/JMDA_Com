"""Prepare and launch the explicitly allocated three-weather few-shot queue."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weather.fewshot_presets import ALLOCATION
from weather.server_adaptive_presets import RECIPE as SERVER_RECIPE
from weather.class_matching import FEWSHOT_PROTOCOL
from tools.run_label_guided_matrix import build_jobs


def build_plan(machine,output,lists,snapshot,seeds=tuple(range(42,47)),fewshot_recipe='baseline'):
    if machine not in ALLOCATION:raise ValueError('Unknown machine allocation')
    if fewshot_recipe not in ('baseline',SERVER_RECIPE):raise ValueError('Unknown few-shot recipe')
    if fewshot_recipe==SERVER_RECIPE and machine!='server':raise ValueError('The adaptive recipe is server-only')
    if not seeds or len(set(seeds))!=len(seeds) or any(seed<0 for seed in seeds):raise ValueError('Seeds must be unique nonnegative integers')
    pairs=ALLOCATION[machine]
    jobs=build_jobs(Path(output),Path(lists),pairs,seeds,Path(snapshot),True,
                    evaluation_mode='validation_ce',unified_fewshot=True,fewshot_recipe=fewshot_recipe)
    targets=sorted({j['values']['target'] for j in jobs})
    return {'machine_split':machine,'protocol':FEWSHOT_PROTOCOL,'pairs':list(pairs),'seeds':list(seeds),
        'epochs':40,'early_stopping_patience':0,'target_labels_per_class':3,
        'fewshot_recipe':fewshot_recipe,'matched_alignment_weight':.2 if fewshot_recipe==SERVER_RECIPE else .1,
        'rainy_distillation_disabled':True,'target_classifier_CE_used':False,'heldout_evaluation':False,
        'total':len(jobs),'counts_by_weather':{t:sum(j['values']['target']==t for j in jobs) for t in targets},
        'jobs':jobs}


def required_weights(plan):
    names=set()
    for job in plan['jobs']:
        a=job['values'];names.add(a['image_backbone'])
        if a.get('ir_backbone'):names.add(a['ir_backbone'])
    return sorted(names)


def check_environment(plan,data_root):
    if not Path(data_root).is_dir():raise FileNotFoundError(data_root)
    from tools.run_backlight_rainy_suite import dependency_preflight
    from tools.download_official_uda_weights import require_weight
    dependency_preflight('rainy')
    for name in required_weights(plan):require_weight(name,ROOT/'pretrained_weights')
    print('[PREFLIGHT PASS] allocation-specific weights, imports, CUDA and data root',flush=True)


def prepare_data(machine,data_root,prepared,snapshot):
    from tools.prepare_weather_generalization_split import prepare
    from weather.train_vds_core import prepare_datasets,paired_epoch_samplers
    from weather.fewshot import TargetLabelBudget
    prepared=Path(prepared);prepared.mkdir(parents=True,exist_ok=False)
    targets=sorted({p.rsplit('_',1)[1] for p in ALLOCATION[machine]})
    env=dict(os.environ,JMDA_DATA_ROOT=str(Path(data_root).resolve()),PYTHONUNBUFFERED='1')
    subprocess.run([sys.executable,'-B',str(Path(snapshot)/'tools/prepare_paired_protocol_lists.py'),
        '--data-root',str(data_root),'--output-dir',str(prepared/'base'),'--targets',*targets,'--modality','paired'],
        env=env,cwd=ROOT,check=True)
    prepare(prepared/'base',prepared/'locked',targets,.2,20261115)
    admissions={}
    for target in targets:
        datasets,audit=prepare_datasets(prepared/'locked',target,True)
        source,domain,*_=datasets
        budget=TargetLabelBudget.from_dataset(domain,audit['compact_label_map'],3,42)
        budget.steps=paired_epoch_samplers(source,domain,16,16,42,'dann')[0];budget.batch_sizes=(16,16)
        m=budget.metadata(.1)
        admissions[target]={'source_train':len(source),'target_train':len(domain),'target_val':len(datasets[-1]),
            'labelled_target_samples':m['labelled_target_samples'],'unlabelled_target_samples':m['unlabelled_target_samples'],
            'class_count':len(audit['original_class_ids']),'seed42_portable_label_pool_sha256':m['label_pool_portable_sha256'],
            'source_classes_equal_target_classes':audit['source_classes_equal_target_classes']}
        print('FEWSHOT_DATA='+json.dumps({target:admissions[target]}),flush=True)
    (prepared/'admission.json').write_text(json.dumps(admissions,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return prepared/'locked'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine',choices=tuple(ALLOCATION),required=True)
    parser.add_argument('--fewshot-recipe',choices=('baseline',SERVER_RECIPE),default='baseline')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(42,47)))
    parser.add_argument('--data-root',type=Path,default=Path(os.environ.get('JMDA_DATA_ROOT',r'D:\Code\JMDA-Net\Data')))
    parser.add_argument('--output',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--check-environment',action='store_true')
    mode.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.fewshot_recipe==SERVER_RECIPE and args.machine!='server':parser.error('The adaptive recipe is server-only')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    suffix='v2_' if args.fewshot_recipe==SERVER_RECIPE else ''
    output=(args.output or ROOT/'runs/weather_label_guided'/f'{args.machine}_fewshot_{suffix}{stamp}').resolve()
    if not output.is_relative_to(ROOT/'runs/weather_label_guided'):parser.error('Use runs/weather_label_guided for separate results')
    if args.execute and not output.name.startswith(args.machine+'_fewshot_'):
        parser.error('Actual outputs must have the machine_fewshot_ prefix')
    prepared=ROOT/'prepared_lists/weather_fewshot'/f'{args.machine}_{stamp}'
    plan=build_plan(args.machine,output,prepared/'locked',ROOT,args.seeds,args.fewshot_recipe)
    print('Allocation='+json.dumps(plan['counts_by_weather'])+'; total='+str(plan['total']),flush=True)
    if args.check_environment:
        check_environment(plan,args.data_root);return
    if not args.execute and not args.prepare_only:
        if output.exists():raise FileExistsError(output)
        output.mkdir(parents=True);(output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('PLAN_ONLY='+str(output),flush=True);return
    if output.exists():raise FileExistsError(output)
    if args.execute:check_environment(plan,args.data_root)
    from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot
    snapshot=prepare_snapshot(ROOT)
    lists=prepare_data(args.machine,args.data_root,prepared,snapshot)
    if args.prepare_only:print('DATA_PREPARATION_ONLY='+str(prepared),flush=True);return
    command=[sys.executable,'-B','-u',str(ROOT/'tools/run_label_guided_matrix.py'),
        '--unified-fewshot','--machine-split',args.machine,'--data-dir',str(lists),'--pairs',*plan['pairs'],
        '--seeds',*[str(s) for s in args.seeds],'--matched-alignment-weight',str(plan['matched_alignment_weight']),
        '--target-labels-per-class','3','--fewshot-recipe',args.fewshot_recipe,
        '--parameter-profile','reference','--evaluation-mode','validation_ce','--complete-budget',
        '--early-stopping-patience','0','--output',str(output),'--execute']
    if args.machine=='local':command+=['--local','--local-lease-timeout-seconds','0']
    if args.fewshot_recipe==SERVER_RECIPE:command+=['--export-compact-diagnostics']
    env=dict(os.environ,JMDA_DATA_ROOT=str(args.data_root.resolve()),JMDA_DETERMINISTIC='1',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',TORCH_HOME=str(ROOT/'pretrained_weights/torch'))
    print('Run directory='+str(output),flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
