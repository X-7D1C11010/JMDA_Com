"""Monotone source-to-target photometric style, fitted on training images only.

No target instance label or validation image enters the fitted lookup tables.
The transform is pointwise: it changes intensity/color, never pixel positions.
"""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import numpy as np
from PIL import Image


def cohort_signature(pairs):
    def canonical(value):
        parts=list((PureWindowsPath(str(value)) if '\\' in str(value) else Path(value)).parts)
        data=[i for i,p in enumerate(parts) if p.casefold()=='data']
        if data:parts=parts[data[-1]+1:]
        return '/'.join(parts).casefold()
    rows=sorted([canonical(row[0]),canonical(row[1])] for row in pairs)
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode('utf-8')).hexdigest()


def image_histograms(pairs):
    rgb=np.zeros((3,256),dtype=np.int64);ir=np.zeros(256,dtype=np.int64)
    for vis_path,ir_path,*_ in pairs:
        with Image.open(vis_path) as image:
            a=np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR))
        with Image.open(ir_path) as image:
            b=np.asarray(image.convert('L').resize((224,224),Image.Resampling.BILINEAR))
        for c in range(3):rgb[c]+=np.bincount(a[:,:,c].ravel(),minlength=256)
        ir+=np.bincount(b.ravel(),minlength=256)
    return {'rgb':rgb,'nir':ir[None]}


def histogram_lookup(source_hist,target_hist):
    source_hist=np.asarray(source_hist,dtype=np.float64)
    target_hist=np.asarray(target_hist,dtype=np.float64)
    if source_hist.shape!=target_hist.shape or source_hist.ndim!=2 or source_hist.shape[1]!=256:
        raise ValueError('Expected matching channel-by-256 image histograms')
    if (source_hist<0).any() or (target_hist<0).any() or (source_hist.sum(1)<=0).any() or (target_hist.sum(1)<=0).any():
        raise ValueError('Invalid training histogram')
    a=np.cumsum(source_hist,axis=1)/source_hist.sum(1,keepdims=True)
    b=np.cumsum(target_hist,axis=1)/target_hist.sum(1,keepdims=True)
    return np.stack([np.searchsorted(y,x,side='left').clip(0,255)/255. for x,y in zip(a,b)]).astype(np.float32)


def fit_training_style(source_pairs,target_pairs,*,source_partition='source_train',target_partition='target_train'):
    if (source_partition,target_partition)!=('source_train','target_train'):
        raise ValueError('Histogram style only fits training partitions source_train + unlabelled target_train')
    if not source_pairs or not target_pairs:raise ValueError('Empty training cohort')
    a,b=image_histograms(source_pairs),image_histograms(target_pairs)
    return {'kind':'source_histogram_to_unlabelled_target_train_v1',
        'fit_partitions':['source_train','target_train'],'target_instance_labels_used':False,
        'validation_used_for_statistics':False,'fit_view':'PIL bilinear224; same geometry as shared evaluation view of train images',
        'fit_counts':{'source_train':len(source_pairs),'target_train':len(target_pairs)},
        'source_cohort_sha256':cohort_signature(source_pairs),'target_cohort_sha256':cohort_signature(target_pairs),
        'lookups':{key:histogram_lookup(a[key],b[key]).tolist() for key in a}}


class HistogramSourceStyle:
    def __init__(self,stats,source_pairs=None,target_pairs=None):
        if isinstance(stats,(str,Path)):stats=json.loads(Path(stats).read_text(encoding='utf-8'))
        if stats['fit_partitions']!=['source_train','target_train'] or stats['target_instance_labels_used'] is not False or stats['validation_used_for_statistics'] is not False:
            raise ValueError('Style statistics violate training-only UDA protocol')
        for pairs,key in ((source_pairs,'source'),(target_pairs,'target')):
            if pairs is not None and cohort_signature(pairs)!=stats[key+'_cohort_sha256']:
                raise ValueError('Style lookup belongs to a different '+key+' cohort')
        self.stats=stats;self._device_tables={}
        for name,channels in (('rgb',3),('nir',1)):
            table=np.asarray(stats['lookups'][name])
            if table.shape!=(channels,256) or not np.isfinite(table).all() or (table<0).any() or (table>1).any() or (np.diff(table,axis=1)<-1e-7).any():
                raise ValueError('Style lookup must be finite, bounded and monotone')

    def __call__(self,batch,probability=0.,generator=None,strength=.85):
        import torch
        if not 0<=probability<=1 or not 0<=strength<1:raise ValueError('Invalid histogram-style probability/strength')
        record={'source_histogram_probability':probability,'source_histogram_strength':strength,
                'source_histogram_augmented':0,'target_histogram_augmented':False}
        if not probability or not strength:return batch,record
        x=batch['rgb'];active=torch.rand((len(x),1,1,1),device=x.device,generator=generator)<probability
        key=(str(x.device),str(x.dtype))
        if key not in self._device_tables:self._device_tables[key]={k:x.new_tensor(v) for k,v in self.stats['lookups'].items()}
        result=dict(batch)
        for name in ('rgb','nir'):
            image=batch[name]
            mean=(image.new_tensor([.485,.456,.406]).view(1,3,1,1) if name=='rgb' else .5)
            std=(image.new_tensor([.229,.224,.225]).view(1,3,1,1) if name=='rgb' else .5)
            raw=(image*std+mean).clamp(0,1);indices=raw*255
            lo=indices.floor().long();hi=(lo+1).clamp(max=255);fraction=indices-lo
            lookup=self._device_tables[key][name]
            mapped=torch.stack([lookup[c if name=='rgb' else 0][lo[:,c]]*(1-fraction[:,c])+
                                lookup[c if name=='rgb' else 0][hi[:,c]]*fraction[:,c] for c in range(3)],1)
            # A positive identity component keeps each channel map strictly
            # increasing even when a target histogram has flat/empty bins.
            mapped=(1-strength)*raw+strength*mapped
            result[name]=torch.where(active,(mapped-mean)/std,image)
        record['source_histogram_augmented']=int(active.sum())
        return result,record
