"""Train-only manifold diagnostic; validation labels only score predictions.

These scores are never attributed to DANN/CORAL/CDAN/MDD/MCC.
"""
import json
from pathlib import Path
import sys
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT","2")
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.probe_weather_feature_geometry import TrainingGeometry, unit_rows
from tools.classification_metrics import classification_metrics


def main():
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    output=ROOT/"results/goal3h_graph_diagnostic_v2.json"
    rows=[]
    with threadpool_limits(limits=2):
        for target in ("dark","rainy","foggy"):
            for backbone,folder in (("convnext_small","convnext_small_feature_probe"),("resnet18","frozen_feature_probe"),
                                    ("dinov2_cls","dinov2_small_feature_probe"),("clip","clip_feature_probe_cached")):
                with np.load(base/folder/(target+"_features.npz")) as d:
                    source,train,val=[d[k].copy() for k in ("source","target_train","target_val")]
                    sy,vy=d["source_labels"].copy(),d["target_val_labels"].copy()
                width=source.shape[1]//2
                for w in (.1,.5,.9):
                    arrays=[unit_rows(np.concatenate((w*unit_rows(a[:,:width]),(1-w)*unit_rows(a[:,width:])),1)) for a in (source,train,val)]
                    for strategy in ("pooled_pca","domain_center","coral"):
                        geometry=TrainingGeometry(strategy,components=64).fit(*arrays[:2])
                        s,t,v=[geometry.transform(a,domain) for a,domain in zip(arrays,("source","target","target"))]
                        x=np.concatenate((s,t));labels=np.concatenate((sy,np.full(len(t),-1)))
                        for neighbors in (10,30):
                            clf=LabelSpreading(kernel="knn",n_neighbors=neighbors,alpha=.2,max_iter=100).fit(x,labels)
                            with np.errstate(invalid="ignore",divide="ignore"):
                                probabilities=clf.predict_proba(v)
                            reachable=np.isfinite(probabilities).all(1)&(np.nan_to_num(probabilities).sum(1)>0)
                            prediction=clf.classes_[np.nan_to_num(probabilities).argmax(1)]
                            if not bool(reachable.all()):
                                fallback=LogisticRegression(C=1.,class_weight="balanced",max_iter=700).fit(s,sy)
                                prediction[~reachable]=fallback.predict(v[~reachable])
                            score=classification_metrics(vy.tolist(),prediction.tolist(),int(sy.max()+1))
                            rows.append({"target":target,"backbone":backbone,"VIS_weight":w,"geometry":strategy,
                                         "neighbors":neighbors,"unreachable_validation_samples":int((~reachable).sum()),
                                         "unreachable_fallback":"source-only fixed-C1 logistic; never target labels", "metrics":score})
                best=max((r for r in rows if r["target"]==target and r["backbone"]==backbone),key=lambda r:r["metrics"]["accuracy"])
                print("GRAPH_DIAGNOSTIC",json.dumps(best),flush=True)
                output.write_text(json.dumps({"kind":"source_labelled_target_unlabelled_graph_diagnostic",
                    "formal_UDA_method_result":False,"target_train_labels_used":False,"val_fitted":False,
                    "results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
