"""Fit portable source style LUTs; target labels are absent from the input."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.data_paths import exact_ir_partner,remap_data_path
from weather.histogram_style import fit_training_style


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    base=ROOT/'runs/weather_local_screen/balanced_v4_seed42_20261114_005133/prepared_exact_pairs'
    for target in ('dark','rainy','foggy'):
        audit=ROOT/f'runs/weather_local_screen/remaining_weather_v10_calibrated_r18_seed42/dann_{target}_calibrated_r18/dann_{target}_seed42/dataset_audit.json'
        d=json.loads(audit.read_text());support=set(d['original_class_ids']);source=[]
        for line in (base/'sunny_train.txt').read_text(encoding='utf-8').splitlines():
            path,label=line.rsplit(maxsplit=1)
            if int(label) in support:
                vis=remap_data_path(path);ir=exact_ir_partner(vis)
                if not ir:raise FileNotFoundError('Source exact IR mate missing')
                source.append((vis,ir))
        pairs=[tuple(remap_data_path(x) for x in line.split('\t')) for line in (base/f'{target}_train_pairs.tsv').read_text(encoding='utf-8').splitlines() if line]
        if any(len(x)!=2 for x in pairs):raise ValueError('Target style input must be label-free pairs')
        if len(source)!=d['counts']['source_train'] or len(pairs)!=d['counts']['target_train']:raise ValueError('Cohort changed')
        stats=fit_training_style(source,pairs);stats['original_class_ids']=sorted(support)
        (args.output/(target+'.json')).write_text(json.dumps(stats,indent=2),encoding='utf-8')
        print('FITTED',target,stats['fit_counts'],'target_labels=False',flush=True)


if __name__=='__main__':main()
