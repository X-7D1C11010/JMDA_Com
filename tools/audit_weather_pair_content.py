"""Check exact VIS+IR file-content duplicates between weather partitions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def audit(data_dir,targets):
    from weather.train_vds_core import prepare_datasets
    cache={};reports={}
    def digest(path):
        key=str(Path(path).resolve())
        if key not in cache:
            h=hashlib.sha256()
            with open(key,'rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            cache[key]=h.hexdigest()
        return cache[key]
    for weather in targets:
        datasets,metadata=prepare_datasets(data_dir,weather,augment=False)
        partitions={name:ds for name,ds in zip(('source_train','target_train','target_val'),
                    (datasets[0],datasets[1],datasets[4]))}
        tables={name:{tuple(digest(p) for p in row[:2]) for row in ds.dataset.paired_samples}
                for name,ds in partitions.items()}
        overlaps={a+'__'+b:len(tables[a]&tables[b]) for a,b in
                  [('source_train','target_train'),('source_train','target_val'),('target_train','target_val')]}
        reports[weather]={'path_overlap':metadata['cross_partition_path_overlap'],
                          'exact_VIS_and_IR_content_overlap':overlaps,
                          'scope':'SHA256 of both raw image files; not a near-duplicate/scene audit'}
    return {'status':'FAIL' if any(any(r['exact_VIS_and_IR_content_overlap'].values()) for r in reports.values()) else 'PASS',
            'sha256_files_checked':len(cache),'targets':reports}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--targets',nargs='+',default=['dark','foggy'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=audit(args.data_dir,args.targets)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
