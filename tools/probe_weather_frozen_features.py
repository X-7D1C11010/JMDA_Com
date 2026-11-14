"""Source-supervised frozen-feature diagnostic, not a formal UDA result."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from vds2raw.adaptation import AdaptationModel
from tools.classification_metrics import classification_metrics
from weather.train_vds_core import prepare_datasets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--targets", nargs="+", default=["dark", "rainy", "foggy", "backlight"])
    p.add_argument("--backbone", choices=("resnet18", "clip_vit_b16", "convnext_small", "dinov2_vits14", "resnet50_ibn_a"), default="resnet18")
    p.add_argument("--target-ir-invert", action="store_true", help="Diagnostic only: invert target thermal polarity")
    p.add_argument("--calibrate-pretrained-ir", action="store_true",
                   help="ConvNeXt diagnostic: absorb ImageNet statistics into the IR stem, keeping loader inputs unchanged")
    p.add_argument("--pooling", choices=("average", "gem3"), default="average",
                   help="Diagnostic only; changes encoder spatial aggregation, not image transforms")
    p.add_argument("--geometry", choices=("jmda_square", "letterbox"), default="jmda_square",
                   help="Diagnostic only; aspect-preserving letterboxing is not the existing training default")
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.backbone in {"clip_vit_b16", "dinov2_vits14"} and args.pooling != "average":
        p.error("GeM is only defined for the CNN feature map")
    if args.calibrate_pretrained_ir and args.backbone != "convnext_small":
        p.error("IR stem calibration is defined only for ConvNeXt's unpadded patch stem")
    torch.set_num_threads(2)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Feature probe output is not empty: {output}")
    device = torch.device(args.device)
    if args.backbone == "resnet18":
        weights = args.weights or str(ROOT/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth")
        model = AdaptationModel("deepcoral", weights, True, use_ais=False, ir_channels=3).to(device).eval().requires_grad_(False)
        if args.pooling == "gem3":
            from weather.spatial_pooling import FixedGeM
            model.rgb_encoder.avgpool = FixedGeM(3.)
            model.nir_encoder.avgpool = FixedGeM(3.)
        def encode(value, key):
            return (model.rgb_encoder if key == "rgb" else model.nir_encoder)(value)
    elif args.backbone == "convnext_small":
        from torchvision.models import convnext_small
        weights = Path(args.weights or Path.home()/".cache/torch/hub/checkpoints/convnext_small-0c510722.pth")
        if not hashlib.sha256(weights.read_bytes()).hexdigest().startswith("0c510722"):
            raise ValueError("Unexpected ConvNeXt-Small ImageNet checkpoint")
        model = convnext_small(weights=None)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.classifier[-1] = torch.nn.Identity()
        if args.pooling == "gem3":
            from weather.spatial_pooling import FixedGeM
            model.avgpool = FixedGeM(3.)
        model = model.to(device).eval().requires_grad_(False)
        ir_model = None
        if args.calibrate_pretrained_ir:
            from copy import deepcopy
            from weather.pretrained_ir import calibrate_half_normalized_stem
            ir_model = deepcopy(model)
            calibrate_half_normalized_stem(ir_model.features[0][0])
        def encode(value, key):
            return (ir_model if key == "nir" and ir_model is not None else model)(value)
    elif args.backbone == "resnet50_ibn_a":
        from weather.ibn_probe import resnet50_ibn_a_probe
        weights = Path(args.weights or ROOT/"pretrained_weights/resnet50_ibn_a-d9d0bb7b.pth")
        if not hashlib.sha256(weights.read_bytes()).hexdigest().startswith("d9d0bb7b"):
            raise ValueError("Unexpected official ResNet50 IBN-a checkpoint")
        model = resnet50_ibn_a_probe()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        model.fc = torch.nn.Identity()
        if args.pooling == "gem3":
            from weather.spatial_pooling import FixedGeM
            model.avgpool = FixedGeM(3.)
        model = model.to(device).eval().requires_grad_(False)
        def encode(value, key):
            return model(value)
    elif args.backbone == "dinov2_vits14":
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn
        weights = Path(args.weights or ROOT/"pretrained_weights/dinov2_vits14_pretrain.pth")
        if not hashlib.sha256(weights.read_bytes()).hexdigest().startswith("b938bf1b"):
            raise ValueError("Unexpected official DINOv2 ViT-S/14 checkpoint")
        model = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, img_size=224)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
        model = model.to(device).eval().requires_grad_(False)
        def encode(value, key):
            return model(value)
    else:
        weights = Path(args.weights or ROOT/"pretrained_weights/clip/ViT-B-16.pt")
        if not hashlib.sha256(weights.read_bytes()).hexdigest().startswith("5806e77c"):
            raise ValueError("Unexpected CLIP image encoder checkpoint")
        model = torch.jit.load(str(weights), map_location=device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        def encode(value, key):
            # Match the frozen encoder's documented image statistics while
            # preserving the same geometric validation view and image inputs.
            mean = value.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
            std = value.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
            image = value*std+mean if key == "rgb" else value*.5+.5
            cm = value.new_tensor([.48145466, .4578275, .40821073]).view(1, 3, 1, 1)
            cs = value.new_tensor([.26862954, .26130258, .27577711]).view(1, 3, 1, 1)
            return model.encode_image((image-cm)/cs).float()
    cache, results = {}, []

    def features(view):
        missing = [i for i, path in enumerate(view.paths) if path not in cache]
        loader = DataLoader(Subset(view, missing), batch_size=8, num_workers=0)
        with torch.no_grad():
            for b in loader:
                rgb = encode(b["rgb"].to(device), "rgb").cpu().numpy()
                nir = b["nir"].to(device)
                if args.target_ir_invert and view.partition != "source_train":
                    nir = -nir
                ir = encode(nir, "nir").cpu().numpy()
                for path, a, c in zip(b["path"], rgb, ir):
                    cache[path] = np.concatenate((a, c))
        return np.stack([cache[path] for path in view.paths])

    for target_name in args.targets:
        datasets, audit = prepare_datasets(args.data_dir, target_name, augment=False)
        _, _, source, target, validation = datasets
        if args.geometry == "letterbox":
            from torchvision import transforms
            from weather.image_geometry import LetterboxSquare
            for view in (source, target, validation):
                vis = view.dataset.transform.transforms
                ir = view.dataset.ir_transform.transforms
                view.dataset.transform = transforms.Compose([LetterboxSquare(224, (124, 116, 104)), *vis[1:]])
                view.dataset.ir_transform = transforms.Compose([LetterboxSquare(224, (128, 128, 128)), *ir[1:]])
        xs, xt, xv = features(source), features(target), features(validation)
        ys = np.asarray([label for _, label in source.dataset.samples])
        yv = np.asarray([label for _, label in validation.dataset.samples])
        # The label-free target view never exports instance labels.
        assert not target.labelled
        np.savez_compressed(output/f"{target_name}_features.npz", source=xs, target_train=xt,
                            target_val=xv, source_labels=ys, target_val_labels=yv)
        branch_width = xs.shape[1]//2
        for mode, sl in [("VIS", slice(0, branch_width)), ("IR", slice(branch_width, None)), ("concat", slice(None))]:
            def normalize(x):
                x = x[:, sl].copy()
                if mode == "concat":
                    for part in [slice(0, branch_width), slice(branch_width, None)]:
                        x[:, part] /= np.maximum(np.linalg.norm(x[:, part], axis=1, keepdims=True), 1e-8)
                return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
            a, b = normalize(xs), normalize(xv)
            with threadpool_limits(limits=2):
                for C in (.1, 1., 10., 100.):
                    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=500, tol=1e-5).fit(a, ys)
                    metrics = classification_metrics(yv.tolist(), clf.predict(b).tolist(), len(audit["original_class_ids"]))
                    results.append({"target": target_name, "mode": mode, "classifier": "logistic", "C": C, "metrics": metrics})
                for C in (1., 10., 100.):
                    clf = SVC(C=C, class_weight="balanced").fit(a, ys)
                    metrics = classification_metrics(yv.tolist(), clf.predict(b).tolist(), len(audit["original_class_ids"]))
                    results.append({"target": target_name, "mode": mode, "classifier": "RBF_SVC", "C": C, "metrics": metrics})
            best = max([r for r in results if r["target"] == target_name and r["mode"] == mode], key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
            print(json.dumps(best), flush=True)
        (output/"diagnostic.json").write_text(json.dumps({"kind": "source_only_frozen_pretrained_feature_probe",
            "backbone": args.backbone,
            "pooling": args.pooling,
            "geometry": args.geometry,
            "weights_file": str(Path(weights).resolve()),
            "weights_SHA256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            "target_ir_invert": args.target_ir_invert,
            "calibrate_pretrained_ir": args.calibrate_pretrained_ir,
            "formal_UDA_result": False, "selection_partition": "target_val", "target_training_labels_used": False,
            "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
