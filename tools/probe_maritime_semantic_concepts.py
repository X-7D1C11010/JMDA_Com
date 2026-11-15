"""Source-fitted semantic concept diagnostic; no target instance labels fit.

The shared vocabulary is fixed before any scoring. Classes remain dataset
numeric labels. No manual mapping from validation classes to concepts exists.
This is an independent CLIP-assisted diagnostic, never a five-method result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from weather.statistical_calibration import unit,fused_features,TrainFeatureCalibration
from tools.classification_metrics import classification_metrics


# Generic vessel types and structural concepts; shared by every weather.
# No weather, scene, vessel ID, class index or validation-derived description.
TYPES=["oil tanker","chemical tanker","gas tanker","bulk carrier","container ship","general cargo ship",
       "dry cargo ship","cargo barge","open cargo barge","covered cargo barge","river barge","flat deck barge",
       "crane barge","construction barge","dredger","tugboat","pushboat","pusher tug","workboat",
       "ferry","passenger ship","motor yacht","sailing yacht","sailing catamaran","small motorboat",
       "speedboat","rigid inflatable boat","fishing trawler","fishing boat","patrol boat","police boat",
       "rescue vessel","rescue lifeboat","pilot boat","fire boat","survey ship","supply vessel","landing craft"]
ATTRIBUTES=["a ship with round tanks on its deck","a tanker with pipes on its deck",
    "a ship with large open cargo holds","a cargo ship with flat hatch covers","a ship carrying containers",
    "a low flat working barge","a ship carrying cranes","a short ship with a tall wheelhouse",
    "a long narrow ship with a flat cargo deck","a ship with a cabin near its stern",
    "a ship with a cabin near its bow","a boat with fishing nets and deck equipment",
    "a boat with masts and sails","a small boat with an outboard motor","a vessel with two parallel hulls",
    "a passenger vessel with multiple decks","a small boat with an enclosed cabin"]


def text_features(device):
    vocabulary=TYPES+ATTRIBUTES
    prompts=[[f"a photo of a {name}.",f"a side view of a {name}.",f"an overhead photo of a {name}."]
             for name in TYPES]+[[f"a photo of {name}.",f"a side view of {name}.",f"an overhead photo of {name}."] for name in ATTRIBUTES]
    signature=hashlib.sha256(json.dumps(prompts,ensure_ascii=True).encode()).hexdigest()
    path=ROOT/"results/maritime_source_concept_text_features.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if z["prompt_sha256"].item()!=signature:raise ValueError("Concept cache vocabulary mismatch")
            return vocabulary,z["vectors"].copy(),signature
    weights=ROOT/"pretrained_weights/clip/ViT-B-16.pt"
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest!="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f":
        raise ValueError("Unexpected CLIP pretrained checkpoint")
    import clip
    model,_=clip.load(str(weights),device=device,jit=False)
    model.eval().requires_grad_(False)
    tokens=clip.tokenize([s for group in prompts for s in group]).to(device)
    pieces=[]
    with torch.no_grad():
        for batch in tokens.split(24):pieces.append(model.encode_text(batch).float().cpu().numpy())
    vectors=unit(unit(np.concatenate(pieces)).reshape(len(vocabulary),3,-1).mean(1))
    np.savez_compressed(path,vectors=vectors,vocabulary=np.asarray(vocabulary),prompt_sha256=np.asarray(signature),
        checkpoint_sha256=np.asarray(digest))
    print("PRETRAINED_TEXT_CONCEPTS",len(vocabulary),"prompt_sha256",signature,flush=True)
    return vocabulary,vectors,signature


def assignment(scores,clusters):
    classes=scores.shape[1]
    matrix=np.stack([scores[clusters==c].mean(0) for c in range(classes)])
    a,b=linear_sum_assignment(-matrix);mapping=np.empty(classes,dtype=np.int64);mapping[a]=b
    return mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2)
    vocabulary,text,prompt_sha=text_features(args.device)
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    rows=[];source_concepts={}
    for name in ("rainy","foggy","dark"):
        with np.load(base/"clip_feature_probe_cached"/f"{name}_features.npz",allow_pickle=False) as z:
            if "target_train_labels" in z.files:raise ValueError("Target labels in semantic diagnostic")
            raw=[z[k].copy() for k in ("source","target_train","target_val")]
            labels,val_labels=z["source_labels"].copy(),z["target_val_labels"].copy()
        classes=int(labels.max())+1
        modalities=[[unit(x[:,:512])@text.T,unit(x[:,512:])@text.T] for x in raw]
        # Relative object-concept scores remove a sample's generic boat/photo affinity.
        modalities=[[x-x.mean(1,keepdims=True) for x in view] for view in modalities]
        source_concepts[name]={str(c):[vocabulary[i] for i in np.argsort(
            modalities[0][0][labels==c].mean(0)-modalities[0][0].mean(0))[-5:][::-1]] for c in range(classes)}
        with np.load(base/"frozen_feature_probe"/f"{name}_features.npz",allow_pickle=False) as z:
            if not np.array_equal(labels,z["source_labels"]) or not np.array_equal(val_labels,z["target_val_labels"]):
                raise ValueError("Encoder cache cohort mismatch")
            main=[torch.tensor(fused_features(z[k],.94),dtype=torch.float32) for k in ("source","target_train","target_val")]
        cal=TrainFeatureCalibration.fit(main[0].numpy(),main[1].numpy(),16)
        with torch.no_grad():t=cal(main[1],"target").numpy();v=cal(main[2],"target").numpy()
        with threadpool_limits(limits=2):km=KMeans(n_clusters=classes,n_init=10,random_state=42).fit(t)
        val_clusters=km.predict(v)
        for target_center in (False,True):
            scaled=[[],[],[]]
            for modality in (0,1):
                a,b,v=[x[modality] for x in modalities]
                scaler=StandardScaler().fit(a)
                if target_center:
                    shift=a.mean(0)-b.mean(0);b=b+shift;v=v+shift
                for container,values in zip(scaled,(a,b,v)):container.append(scaler.transform(values))
            for weight in (.75,.5,.25):
                source,train,val=[np.concatenate((weight*x[0],(1-weight)*x[1]),1) for x in scaled]
                for strength in (.1,1.,10.):
                    with threadpool_limits(limits=2):
                        clf=LogisticRegression(C=strength,class_weight="balanced",max_iter=2000).fit(source,labels)
                    settings={"target":name,"target_training_mean_centering":target_center,"VIS_weight":weight,"C":strength}
                    pred=clf.predict(val)
                    rows.append({**settings,"evaluation":"source_semantic_classifier",
                        "seen_source_fit_ACC":float(np.mean(clf.predict(source)==labels)),
                        "metrics":classification_metrics(val_labels.tolist(),pred.tolist(),classes)})
                    scores=clf.decision_function(train)
                    for standardized in (False,True):
                        if standardized:scores=scores/np.maximum(clf.decision_function(source).std(0),1e-6)
                        mapping=assignment(scores,km.labels_)
                        rows.append({**settings,"evaluation":"fixed_R18_geometry_semantic_source_scores",
                            "source_score_standardization":standardized,"mapping":mapping.tolist(),
                            "metrics":classification_metrics(val_labels.tolist(),mapping[val_clusters].tolist(),classes)})
        best=max((r for r in rows if r["target"]==name),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
        print(json.dumps({"target":name,"best":best}),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({"kind":"source_fitted_pretrained_semantic_concepts_diagnostic",
            "formal_DA_method_result":False,"target_training_labels_used":False,"validation_used_for_fitting":False,
            "validation_labels_used_only_for_scoring":True,"vocabulary_shared_across_weather":True,
            "vocabulary":vocabulary,"prompt_sha256":prompt_sha,"source_class_top_concepts":source_concepts,
            "no_dataset_classes_manually_renamed_or_swapped":True,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
