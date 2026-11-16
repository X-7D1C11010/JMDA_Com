"""Create fixed exact-pair-grouped retraining holdouts without moving images.

Historical experiments used the complete target_train. This is a new locked
retraining split, not a claim that its samples were historically unseen.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.uda_data import read_image_list
from tools.data_paths import remap_data_path,exact_ir_partner


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare(input_dir,output_dir,targets=('dark','foggy'),fraction=.2,seed=20261115):
    input_dir,output_dir=Path(input_dir).resolve(),Path(output_dir).resolve()
    if not 0<fraction<.5:raise ValueError('Holdout fraction must lie in (0,0.5)')
    if output_dir.exists():raise FileExistsError(output_dir)
    payload={'sunny_train.txt':(input_dir/'sunny_train.txt').read_bytes().decode('utf-8')}
    reports={};hashes={}
    def pair_hash(pair):
        result=[]
        for path in pair:
            if path not in hashes:hashes[path]=digest(path)
            result.append(hashes[path])
        return tuple(result)
    def listed_groups(name,support):
        result=set()
        for path,label in read_image_list(input_dir/name):
            if label not in support:continue
            vis=str(Path(remap_data_path(path)).resolve());ir=exact_ir_partner(vis)
            if not ir:raise FileNotFoundError(f'Missing infrared mate: {vis}')
            result.add(pair_hash((vis,str(Path(ir).resolve()))))
        return result
    for target in targets:
        rows=read_image_list(input_dir/f'{target}_train.txt')
        rawpairs=[line.split('\t') for line in (input_dir/f'{target}_train_pairs.tsv').read_text(encoding='utf-8').splitlines() if line.strip()]
        pairs={str(Path(remap_data_path(row[0])).resolve()):tuple(str(Path(remap_data_path(p)).resolve()) for p in row[:2]) for row in rawpairs}
        normalized=[(str(Path(remap_data_path(p)).resolve()),y) for p,y in rows]
        if len(pairs)!=len(rows) or {p for p,y in normalized}!=set(pairs):raise ValueError('Label/pair cohort mismatch')
        groups=defaultdict(lambda:defaultdict(list));group_labels={}
        for path,label in normalized:
            key=pair_hash(pairs[path])
            if key in group_labels and group_labels[key]!=label:raise ValueError('Identical image pair has conflicting labels')
            group_labels[key]=label;groups[label][key].append(path)
        if not set(groups)<={y for p,y in read_image_list(input_dir/'sunny_train.txt')}:
            raise ValueError('A target class has no source training samples')
        if set(groups)!={y for p,y in read_image_list(input_dir/f'{target}_val.txt')}:
            raise ValueError('Target training/validation class support differs')
        source_groups=listed_groups('sunny_train.txt',set(groups))
        val_groups=listed_groups(f'{target}_val.txt',set(groups))
        original_groups=set(group_labels)
        if original_groups&source_groups or original_groups&val_groups or source_groups&val_groups:
            raise ValueError('Exact VIS/IR pair content overlaps original train/validation partitions')
        selected=set()
        for label,table in sorted(groups.items()):
            keys=sorted(table)
            if len(keys)<2:raise ValueError(f'{target} class {label} has fewer than two distinct image-pair groups')
            random.Random(seed+100003*label).shuffle(keys)
            desired=max(1,round(sum(map(len,table.values()))*fraction));count=0
            for key in keys[:-1]:
                selected.update(table[key]);count+=len(table[key])
                if count>=desired:break
        train=[row for row in normalized if row[0] not in selected]
        test=[row for row in normalized if row[0] in selected]
        if {y for p,y in train}!={y for p,y in test}:raise ValueError('Split class support differs')
        def labels(data):return ''.join(f'{path} {label}\n' for path,label in data)
        def pair_list(data):return ''.join('\t'.join(pairs[path])+'\n' for path,label in data)
        payload[f'{target}_train.txt']=labels(train)
        payload[f'{target}_train_pairs.tsv']=pair_list(train)
        payload[f'{target}_test.txt']=labels(test)
        payload[f'{target}_test_pairs.tsv']=pair_list(test)
        payload[f'{target}_val.txt']=(input_dir/f'{target}_val.txt').read_bytes().decode('utf-8')
        val_paths={str(Path(remap_data_path(p)).resolve()) for p,y in read_image_list(input_dir/f'{target}_val.txt')}
        if val_paths & set(pairs):raise ValueError('Original train/validation paths overlap')
        reports[target]={'original_target_train':len(rows),'target_train':len(train),'target_val':len(val_paths),
            'target_test':len(test),'train_class_counts':dict(Counter(y for p,y in train)),
            'test_class_counts':dict(Counter(y for p,y in test)),
            'source_or_original_val_exact_pair_content_overlap':0,
            'test_pair_group_sha256':hashlib.sha256(json.dumps(sorted(
                (label,pair_hash(pairs[path])) for path,label in test),sort_keys=True).encode('utf-8')).hexdigest(),
            'train_test_exact_pair_content_group_overlap':0,'all_original_train_samples_accounted_for':len(train)+len(test)==len(rows)}
    manifest={'protocol':'LOCKED_RETRAIN_HOLDOUT_V1','split_seed':seed,'requested_holdout_fraction':fraction,
        'grouping':'exact VIS+IR file SHA256 pair; not scene/MMSI grouping',
        'historical_experiments_used_the_complete_target_train':True,'historically_unseen_test':False,
        'test_for_future_retraining_only':'excluded from gradients, fitted statistics and model/hyperparameter selection',
        'target_val_unchanged':True,'targets':reports,
        'input_list_sha256':{name:digest(input_dir/name) for name in payload if (input_dir/name).is_file()},
        'output_list_sha256':{name:hashlib.sha256(value.encode('utf-8')).hexdigest() for name,value in payload.items()}}
    output_dir.mkdir(parents=True)
    for name,value in payload.items():
        with (output_dir/name).open('w',encoding='utf-8',newline='\n') as stream:stream.write(value)
    (output_dir/'holdout_protocol_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--targets',nargs='+',default=['dark','foggy'],choices=['dark','foggy','rainy','backlight'])
    p.add_argument('--test-fraction',type=float,default=.2);p.add_argument('--split-seed',type=int,default=20261115)
    args=p.parse_args();print(json.dumps(prepare(args.input_dir,args.output_dir,args.targets,args.test_fraction,args.split_seed),ensure_ascii=False))


if __name__=='__main__':main()
