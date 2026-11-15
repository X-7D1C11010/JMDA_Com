"""Source-fitted IR normalization diagnostic; no target labels in fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torchvision.models import convnext_small
from torch.utils.data import DataLoader,Subset
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from weather.ir_input_normalizer import InfraredInputNormalizer
from weather.train_vds_core import prepare_datasets
from tools.classification_metrics import classification_metrics


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42)
    weight=Path.home()/'.cache/torch/hub/checkpoints/convnext_small-0c510722.pth'
    if not hashlib.sha256(weight.read_bytes()).hexdigest().startswith('0c510722'):raise ValueError('Wrong pretrained checkpoint')
    model=convnext_small(weights=None);model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
    model.classifier[-1]=torch.nn.Identity();model.cuda().eval().requires_grad_(False)
    base=ROOT/'runs/weather_local_screen/balanced_v4_seed42_20261114_005133'
    rows=[]
    for mode in ('zscore','percentile'):
        for imagenet in (False,True):
            op=InfraredInputNormalizer(mode,imagenet).cuda().eval();cache={}
            def features(ds):
                missing=[i for i,path in enumerate(ds.paths) if path not in cache]
                with torch.no_grad():
                    for batch in DataLoader(Subset(ds,missing),batch_size=8,num_workers=0,generator=torch.Generator().manual_seed(0)):
                        vectors=model(op(batch['nir'].cuda())).cpu().numpy()
                        for path,v in zip(batch['path'],vectors):cache[path]=v
                return np.stack([cache[path] for path in ds.paths])
            for weather in ('dark','rainy','foggy'):
                _,_,source,_,validation=prepare_datasets(base/'prepared_exact_pairs',weather,False)[0]
                si,vi=features(source),features(validation)
                with np.load(base/f'convnext_small_feature_probe/{weather}_features.npz') as f:
                    sv,vv=f['source'][:,:768].copy(),f['target_val'][:,:768].copy()
                    ys,yv=f['source_labels'].copy(),f['target_val_labels'].copy()
                if not np.array_equal(ys,[y for _,y in source.dataset.samples]):raise ValueError('Source cohort changed')
                packed={'source_VIS':sv,'source_IR':si,'target_val_VIS':vv,'target_val_IR':vi,'source_labels':ys,'target_val_labels':yv}
                np.savez_compressed(args.output/f'{weather}_{mode}_imagenet{imagenet}.npz',**packed)
                with threadpool_limits(limits=2):
                    for w in (0.,.1,.5,.9):
                        a=unit(np.concatenate((w*unit(sv),(1-w)*unit(si)),1));b=unit(np.concatenate((w*unit(vv),(1-w)*unit(vi)),1))
                        for C in (1.,10.,100.):
                            clf=LogisticRegression(C=C,class_weight='balanced',max_iter=700,random_state=42).fit(a,ys)
                            metrics=classification_metrics(yv.tolist(),clf.predict(b).tolist(),int(ys.max()+1))
                            rows.append({'target':weather,'IR_mode':mode,'ImageNet_IR_statistics':imagenet,'VIS_weight':w,'C':C,'metrics':metrics})
                best=max((r for r in rows if r['target']==weather and r['IR_mode']==mode and r['ImageNet_IR_statistics']==imagenet),key=lambda r:(r['metrics']['accuracy'],r['metrics']['f1_macro']))
                print('IR_NORMALIZATION_DIAGNOSTIC',json.dumps(best),flush=True)
                (args.output/'diagnostic.json').write_text(json.dumps({'formal_UDA_method_result':False,
                    'target_instance_labels_used_for_fitting':False,'validation_set_statistics_fitted':False,
                    'normalization_scope':'per image in encoder; independent of other batch members; no running statistics',
                    'results':rows},indent=2),encoding='utf-8')


if __name__=='__main__':main()
