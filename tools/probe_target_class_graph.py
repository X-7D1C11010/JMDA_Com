"""Source-labelled / target-unlabelled class-graph association diagnostic.

Exact enumeration is bounded to <=9 classes. Validation labels never enter
the graph fit or assignment. This is not a named method's training result.
QAP formulation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.quadratic_assignment.html
"""
from itertools import islice,permutations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.probe_weather_feature_geometry import TrainingGeometry,unit_rows
from tools.classification_metrics import classification_metrics


def match_graphs(source_graph,target_graph,cross_cost,gammas=(0.,.25,.5,.75,1.)):
    a,b,c=[np.asarray(v,dtype=np.float64) for v in (source_graph,target_graph,cross_cost)]
    n=len(a)
    if (not 2<=n<=9 or a.shape!=(n,n) or b.shape!=a.shape or c.shape!=a.shape
            or not all(np.isfinite(v).all() for v in (a,b,c))
            or any(not 0<=g<=1 for g in gammas)):
        raise ValueError('Invalid bounded class-graph assignment')
    a=a/max(np.linalg.norm(a),1e-12);b=b/max(np.linalg.norm(b),1e-12)
    c=c/max(float(c.mean()),1e-12)
    best=[None]*len(gammas);scores=np.full(len(gammas),np.inf)
    iterator=permutations(range(n))
    while True:
        chunk=list(islice(iterator,4096))
        if not chunk:break
        p=np.asarray(chunk,dtype=np.int64)
        graph_error=((a[None]-b[p[:,:,None],p[:,None,:]])**2).sum((1,2))
        cross_error=c[np.arange(n)[None],p].mean(1)
        for i,gamma in enumerate(gammas):
            objective=(1-gamma)*graph_error+gamma*cross_error;j=int(objective.argmin())
            if objective[j]<scores[i]:
                scores[i]=objective[j];best[i]=p[j].copy()
    result=[]
    for gamma,score,p in zip(gammas,scores,best):
        cluster_to_class=np.empty(n,dtype=np.int64);cluster_to_class[p]=np.arange(n)
        result.append({'gamma':gamma,'fit_objective':float(score),'cluster_to_source_class':cluster_to_class.tolist()})
    return result


def fit_association(source,source_labels,target,components):
    classes=len(np.unique(source_labels))
    if set(source_labels.tolist())!=set(range(classes)):raise ValueError('Expected shared compact source class IDs')
    dim=min(components,source.shape[1],len(source)-1,len(target)-1)
    source_projection=PCA(n_components=dim,svd_solver='full').fit(source)
    target_projection=PCA(n_components=dim,svd_solver='full').fit(target)
    s,t=source_projection.transform(source),target_projection.transform(target)
    clustering=KMeans(n_clusters=classes,n_init=20,random_state=42).fit(t)
    prototypes=np.stack([s[source_labels==label].mean(0) for label in range(classes)])
    source_graph=cdist(prototypes,prototypes)
    target_graph=cdist(clustering.cluster_centers_,clustering.cluster_centers_)
    common=TrainingGeometry('coral',components=dim).fit(source,target)
    sx,tx=common.transform(source,'source'),common.transform(target,'target')
    sp=unit_rows(np.stack([sx[source_labels==label].mean(0) for label in range(classes)]))
    tp=unit_rows(np.stack([tx[clustering.labels_==cluster].mean(0) for cluster in range(classes)]))
    mappings=match_graphs(source_graph,target_graph,cdist(sp,tp))
    return target_projection,clustering,mappings


def main():
    base=ROOT/'runs/weather_local_screen/balanced_v4_seed42_20261114_005133'
    cases=[]
    for target in ('dark','foggy'):
        for name,folder in (('resnet18','frozen_feature_probe'),('convnext_small','convnext_small_feature_probe')):
            cases.append((target,name,base/folder/(target+'_features.npz'),None))
    selected=ROOT/'runs/weather_enhanced_screen/ir_statistics_mcc_foggy_seed42_v1/mcc_foggy_source0.1_target0.1/mcc_foggy_seed42'
    cases.append(('foggy','selected_MCC',selected/'selected_feature_cache.npz',768))
    results=[]
    for target,name,path,width in cases:
        with np.load(path,allow_pickle=False) as packed:
            if 'target_train_labels' in packed.files:raise RuntimeError('Target training truth is forbidden')
            source,target_train=packed['source'].copy(),packed['target_train'].copy()
            source_labels=packed['source_labels'].copy()
        width=width or source.shape[1]//2
        for weight in (.1,.5,.9):
            def features(x):return unit_rows(np.concatenate((weight*unit_rows(x[:,:width]),(1-weight)*unit_rows(x[:,width:])),1))
            s,t=features(source),features(target_train)
            projection,clustering,mappings=fit_association(s,source_labels,t,32 if target=='dark' else 16)
            # Only now read validation data/labels; fitting has already ended.
            with np.load(path,allow_pickle=False) as packed:
                validation=features(packed['target_val']);labels=packed['target_val_labels'].tolist()
            clusters=clustering.predict(projection.transform(validation))
            for mapping in mappings:
                prediction=np.asarray(mapping['cluster_to_source_class'])[clusters]
                metrics=classification_metrics(labels,prediction.tolist(),len(np.unique(source_labels)))
                results.append({'target':target,'pack':name,'VIS_weight':weight,**mapping,'metrics':metrics})
        best=max((r for r in results if r['target']==target and r['pack']==name),key=lambda r:r['metrics']['accuracy'])
        print('TRAIN_FITTED_GRAPH_DIAGNOSTIC',target,name,round(100*best['metrics']['accuracy'],2),best['gamma'],best['VIS_weight'],flush=True)
    payload={'kind':'train_fitted_source_class_graph_target_cluster_association','formal_training_result':False,
        'target_training_labels_used':False,'validation_used_for_assignment':False,
        'validation_labels_used_only_for_scoring':True,'assignment_never_uses_validation_label_permutations':True,
        'results':results}
    (ROOT/'results/train_fitted_target_class_graph_seed42.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
