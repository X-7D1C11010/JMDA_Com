"""FDA style screen on labelled-source features; not five-method UDA results.

Each source training crop contributes a clean and a styled view, each weighted
by 1/2. Only unlabelled target_train images supply style. Target validation
features and labels are never passed to fit or statistics estimation.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from tools.classification_metrics import classification_metrics
from tools.reproducibility import seed_everything
from vds2raw.adaptation import AdaptationModel
from weather.fourier_style import augment_source_style
from weather.train_vds_core import prepare_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--vis-weight", type=float, required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=[0., .01, .05])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    if not .05 < args.vis_weight < .95:
        raise ValueError("Use a VIS weight strictly between .05 and .95")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    seed_everything(args.seed)
    sets, audit = prepare_datasets(args.data_dir, args.target, augment=True)
    source, target, _, _, validation = sets
    validation.cache_evaluation_inputs(workers=2)
    classes = len(audit["original_class_ids"])
    model = AdaptationModel("deepcoral", args.weights, True, use_ais=False, ir_channels=3,
                            image_backbone="convnext_small", freeze_image_encoders=True,
                            num_classes=classes, feature_normalization="unit").cuda().eval()
    with torch.no_grad():
        model.modality_logits.copy_(torch.tensor([(args.vis_weight-.05)/.90, (.95-args.vis_weight)/.90],
                                                 device="cuda").log())
    def move(batch):
        return {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    validation_features, validation_labels = [], []
    with torch.no_grad():
        for batch in DataLoader(validation, batch_size=16, num_workers=0,
                                generator=torch.Generator().manual_seed(args.seed+1)):
            validation_features.append(model.encode(move(batch)).cpu().numpy())
            validation_labels.extend(batch["label"].tolist())
    validation_features = np.concatenate(validation_features)
    results = []
    for beta in args.betas:
        source_loader = DataLoader(source, batch_size=16, shuffle=True, num_workers=2,
                                   generator=torch.Generator().manual_seed(args.seed+20000))
        style_loader = DataLoader(target, batch_size=16, shuffle=True, num_workers=2,
                                  generator=torch.Generator().manual_seed(args.seed+30000))
        style_iterator = iter(style_loader)
        features, labels = [], []
        augmentation_generator = torch.Generator(device="cuda").manual_seed(args.seed+33000)
        with torch.no_grad():
            for batch in source_loader:
                try:
                    style = next(style_iterator)
                except StopIteration:
                    style_iterator = iter(style_loader)
                    style = next(style_iterator)
                if "label" in style:
                    raise RuntimeError("Target training labels entered the FDA screen")
                batch, style = move(batch), move(style)
                changed, _ = augment_source_style(batch, style, beta, 1., augmentation_generator)
                for view in (batch, changed):
                    features.append(model.encode(view).cpu().numpy())
                    labels.extend(batch["label"].cpu().tolist())
        features = np.concatenate(features)
        with threadpool_limits(limits=2):
            classifier = LogisticRegression(C=1., class_weight="balanced", max_iter=1000, tol=1e-5)
            classifier.fit(features, labels, sample_weight=np.full(len(labels), .5))
            prediction = classifier.predict(validation_features)
        metrics = classification_metrics(validation_labels, prediction.tolist(), classes)
        result = {"beta": beta, "source_labelled_samples": len(source), "fitted_augmented_rows": len(labels),
                  "per_view_sample_weight": .5, "expected_style_probability": .5, "target_val": metrics}
        results.append(result)
        print(json.dumps(result), flush=True)
        (output/"diagnostic.json").write_text(json.dumps({"kind": "FDA_source_feature_diagnostic",
            "formal_UDA_result": False, "arguments": vars(args), "dataset_audit": audit,
            "target_training_labels_used": False, "validation_used_for_fit": False,
            "FDA_permission": "user explicitly approved additional FDA local single-seed screening",
            "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
