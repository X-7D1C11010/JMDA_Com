"""Photometric source-style diagnostic; never reported as a DA method result."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torchvision.models import convnext_small
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from weather.histogram_style import HistogramSourceStyle
from weather.train_vds_core import prepare_datasets
from tools.classification_metrics import classification_metrics


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42)
    weights=Path.home()/'.cache/torch/hub/checkpoints/convnext_small-0c510722.pth'
    if not hashlib.sha256(weights.read_bytes()).hexdigest().startswith('0c510722'):raise ValueError('Checkpoint mismatch')
    model=convnext_small(weights=None);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    model.classifier[-1]=torch.nn.Identity();model.cuda().eval().requires_grad_(False)
    base=ROOT/'runs/weather_local_screen/balanced_v4_seed42_20261114_005133'
    rows=[];started=time.perf_counter()
    for weather in ('dark','rainy','foggy'):
        _,_,source,target,validation=prepare_datasets(base/'prepared_exact_pairs',weather,False)[0]
        op=HistogramSourceStyle(ROOT/f'runs/weather_local_screen/histogram_style_stats_seed42/{weather}.json',
                               source.dataset.paired_samples,target.dataset.paired_samples)
        with np.load(base/f'convnext_small_feature_probe/{weather}_features.npz') as f:
            original=f['source'].copy();v=f['target_val'].copy();ys=f['source_labels'].copy();yv=f['target_val_labels'].copy()
        if not np.array_equal(ys,np.asarray([y for _,y in source.dataset.samples])):raise ValueError('Source order changed')
        if not np.array_equal(yv,np.asarray([y for _,y in validation.dataset.samples])):raise ValueError('Validation order changed')
        packed={'source_original':original,'target_val':v,'source_labels':ys,'target_val_labels':yv}
        for strength in (.5,.85):
            source_rows=[];generator=torch.Generator(device='cuda').manual_seed(42)
            with torch.no_grad():
                for i,b in enumerate(DataLoader(source,batch_size=8,num_workers=0,generator=torch.Generator().manual_seed(43))):
                    b={k:x.cuda() if torch.is_tensor(x) else x for k,x in b.items()}
                    b,_=op(b,1.,generator,strength=strength)
                    source_rows.append(torch.cat((model(b['rgb']),model(b['nir'])),1).cpu().numpy())
            styled=np.concatenate(source_rows);packed['source_style_'+str(strength)]=styled
            width=styled.shape[1]//2
            with threadpool_limits(limits=2):
                for w in (.1,.5,.9):
                    def normalized(x):return unit(np.concatenate((w*unit(x[:,:width]),(1-w)*unit(x[:,width:])),1))
                    a,b,c=map(normalized,(original,styled,v))
                    for mixture in (.75,1.):
                        x=np.concatenate((a,b)) if mixture<1 else b
                        y=np.concatenate((ys,ys)) if mixture<1 else ys
                        sw=np.r_[np.full(len(ys),1-mixture),np.full(len(ys),mixture)] if mixture<1 else None
                        for C in (1.,10.,100.):
                            clf=LogisticRegression(C=C,class_weight='balanced',max_iter=700,random_state=42).fit(x,y,sample_weight=sw)
                            metrics=classification_metrics(yv.tolist(),clf.predict(c).tolist(),int(ys.max()+1))
                            rows.append({'target':weather,'style_strength':strength,'style_probability':mixture,'VIS_weight':w,'C':C,'metrics':metrics})
            print('HISTOGRAM_FEATURES',weather,strength,flush=True)
        np.savez_compressed(args.output/(weather+'_features.npz'),**packed)
        best=max((r for r in rows if r['target']==weather),key=lambda r:(r['metrics']['accuracy'],r['metrics']['f1_macro']))
        print('BEST_DIAGNOSTIC',json.dumps(best),flush=True)
        (args.output/'diagnostic.json').write_text(json.dumps({'formal_UDA_method_result':False,
            'target_instance_labels_used_for_training':False,'validation_used_for_fitting':False,
            'elapsed_s':time.perf_counter()-started,'results':rows},indent=2),encoding='utf-8')


if __name__=='__main__':main()
