"""Weather-specific feature weights and diagnostics for the pinned UDA trainers."""

from contextlib import contextmanager
from copy import copy
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from historical_uda_metrics import classification_metrics
from historical_uda_extensions import ImageOnlyView
from uda_multimodal import ir_transform_from_vis


def modality_modules(model, method):
    if method == "dann":
        return model.encoder.vis_encoder, model.encoder.ir_encoder
    if method == "deepcoral":
        return model.sharedNet, model.irNet
    if method == "cdan":
        return model.vis, model.ir
    return model.backbone.vis_encoder, model.backbone.ir_encoder


class WeatherFeatureWeights(nn.Module):
    """One bounded VIS weight, shared by source and target for one weather."""

    def __init__(self, method, weather, learn=True, minimum=0.05):
        super().__init__()
        if not 0 <= minimum < 0.5:
            raise ValueError("Minimum modality weight must lie in [0, 0.5)")
        self.logit = nn.Parameter(torch.zeros(()), requires_grad=learn)
        self.minimum = minimum
        self.method = method
        self.weather = weather

    def weights(self):
        vis = self.minimum + (1 - 2 * self.minimum) * self.logit.sigmoid()
        return vis, 1 - vis

    def snapshot(self):
        vis, ir = self.weights()
        vis_value, ir_value = float(vis.detach()), float(ir.detach())
        return {
            "method": self.method, "weather": self.weather,
            "vis": vis_value, "ir": ir_value,
            "larger_weight_modality": ("equal" if vis_value == ir_value else
                                       "VIS" if vis_value > ir_value else "IR"),
            "minimum": self.minimum, "learnable": self.logit.requires_grad,
            "feature_scaling": "VIS=2*w_VIS, IR=2*w_IR before concatenation",
        }


def install_weather_weights(model, method, weather, learn=True):
    gate = WeatherFeatureWeights(method, weather, learn).to(next(model.parameters()).device)
    model.weather_fusion = gate
    for index, module in enumerate(modality_modules(model, method)):
        def weight_output(_module, _inputs, output, branch=index):
            features = output[0] if isinstance(output, tuple) else output
            scale = 2 * gate.weights()[branch]
            diagnostics = getattr(model, "_uda_diagnostics", None)
            if diagnostics is not None:
                diagnostics.record_features(features, branch, scale)
            weighted = features * scale
            return (weighted, *output[1:]) if isinstance(output, tuple) else weighted
        module.register_forward_hook(weight_output)
    return gate


def limited_evaluation_threads(function):
    @wraps(function)
    def evaluate(*args, **kwargs):
        original = torch.get_num_threads()
        try:
            torch.set_num_threads(min(original, 4))
            return function(*args, **kwargs)
        finally:
            torch.set_num_threads(original)
    return evaluate


@contextmanager
def evaluation_state(model, device):
    modes = {module: module.training for module in model.modules()}
    original_threads = torch.get_num_threads()
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" else [])
    try:
        with torch.random.fork_rng(devices=devices):
            torch.set_num_threads(min(original_threads, 4))
            model.eval()
            yield
    finally:
        for module, training in modes.items():
            module.training = training
        torch.set_num_threads(original_threads)


class RunDiagnostics:
    def __init__(self, model, method, args, source_dataset, target_val_dataset,
                 transform, forward, output_dir, *, target_dataset=None, optimizer=None):
        self.model = model
        self.method = method
        self.args = args
        self.device = next(model.parameters()).device
        self.forward = forward
        self.directory = Path(output_dir) / "diagnostics"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source = copy(source_dataset)
        self.source.transform = transform
        self.source.ir_transform = ir_transform_from_vis(transform)
        self.target_train = copy(target_dataset) if target_dataset is not None else None
        if self.target_train is not None:
            self.target_train.transform = transform
            self.target_train.ir_transform = ir_transform_from_vis(transform)
        self.target_val = target_val_dataset
        self.optimizer = optimizer
        self.class_ids = sorted(source_dataset.label_map)
        self.position = 0
        self.domain = None
        self.source_batch_size = 0
        self.partition = None
        self.probabilities = []
        self.features = {}
        self.loss_sums = None
        self.steps = 0
        self.source_seen = self.target_seen = 0
        self.source_seconds = 0.0
        self.weight_history = {}
        self.source_history = {}
        self.target_history = {}
        self.target_prediction_history = {}
        self.gradient_history = []
        self.target_prediction_seconds = 0.0
        head = (model.class_classifier if method == "dann" else
                model.fc if method in {"deepcoral", "cdan"} else model.head)
        head.register_forward_hook(self._capture_predictions)
        model._uda_diagnostics = self
        self._write("fusion_weights.jsonl", {"position": 0, **model.weather_fusion.snapshot()})
        if (optimizer is not None and model.weather_fusion.logit.requires_grad and
                not any(parameter is model.weather_fusion.logit
                        for group in optimizer.param_groups for parameter in group["params"])):
            raise RuntimeError("Learnable weather weight is missing from the optimizer")
        trainer = Path(sys.argv[0])
        self._write("run_configuration.jsonl", {
            "method": method, "weather": args.target, "seed": args.seed,
            "arguments": vars(args), "class_ids_compact_order": self.class_ids,
            "source_sample_count": len(self.source), "target_val_sample_count": len(self.target_val),
            "target_train_sample_count": len(self.target_train) if self.target_train is not None else None,
            "source_manifest_sha256": self.manifest_hash(self.source, labelled=True),
            "target_val_manifest_sha256": self.manifest_hash(self.target_val, labelled=True),
            "target_train_manifest_sha256": (self.manifest_hash(self.target_train, labelled=False)
                                             if self.target_train is not None else None),
            "training_transform": repr(getattr(source_dataset, "transform", None)),
            "evaluation_transform": repr(transform), "model": repr(model),
            "parameters": [{"name": name, "shape": list(parameter.shape),
                            "trainable": parameter.requires_grad}
                           for name, parameter in model.named_parameters()],
            "optimizer_groups": self.optimizer_groups(),
            "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest() if trainer.is_file() else None,
            "diagnostics_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gradient_sampling": "last training batch of each reporting window",
            "fusion_weights": model.weather_fusion.snapshot(),
        })

    @staticmethod
    def manifest_hash(dataset, *, labelled):
        pairs = getattr(dataset, "paired_samples", None)
        rows = pairs if pairs is not None else dataset.samples
        records = [list(row) if labelled else list(row[:2] if pairs is not None else row[:1])
                   for row in rows]
        return hashlib.sha256(json.dumps(records, ensure_ascii=False).encode("utf-8")).hexdigest()

    def optimizer_groups(self):
        if self.optimizer is None:
            return []
        return [{
            "group": index, "lr": group["lr"], "weight_decay": group.get("weight_decay", 0),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "contains_weather_weight": any(parameter is self.model.weather_fusion.logit
                                           for parameter in group["params"]),
        } for index, group in enumerate(self.optimizer.param_groups)]

    def _write(self, filename, record):
        with (self.directory / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def set_domain(self, domain, source_count=0):
        self.domain = domain
        self.source_batch_size = source_count

    def record_features(self, features, branch, scale):
        if self.domain is None:
            return
        parts = ((self.domain, features),)
        if self.domain == "mixed":
            parts = (("source", features[:self.source_batch_size]),
                     ("target", features[self.source_batch_size:]))
        for domain, values in parts:
            norms = values.detach().flatten(1).norm(dim=1)
            statistics = torch.stack((
                norms.sum(), norms.square().sum(), (norms * scale.detach()).sum(),
            ))
            key = (domain, branch)
            old_sum, old_count = self.features.get(key, (0, 0))
            self.features[key] = old_sum + statistics, old_count + len(values)

    def feature_summary(self, domains):
        result = {}
        for domain in domains:
            result[domain] = {}
            for branch, name in enumerate(("VIS", "IR")):
                entry = self.features.pop((domain, branch), None)
                if entry is None:
                    continue
                values, count = entry
                total, squares, weighted_total = values.cpu().tolist()
                average = total / count
                result[domain][name] = {
                    "samples": count, "raw_l2_mean": average,
                    "raw_l2_std": max(squares / count - average**2, 0)**0.5,
                    "weighted_l2_mean": weighted_total / count,
                }
        return result

    def training_step(self, cls_loss, transfer_loss, weighted_loss,
                      source_count, target_count, gradient_sample, position):
        self.position = position
        values = torch.stack((cls_loss.detach(), transfer_loss.detach(), weighted_loss.detach()))
        self.loss_sums = values if self.loss_sums is None else self.loss_sums + values
        self.steps += 1
        self.source_seen += source_count
        self.target_seen += target_count
        if gradient_sample:
            self.record_gradients(cls_loss, weighted_loss)

    def record_gradients(self, classification_loss, weighted_transfer_loss):
        vis, ir = modality_modules(self.model, self.method)
        groups = (("VIS", tuple(p for p in vis.parameters() if p.requires_grad)),
                  ("IR", tuple(p for p in ir.parameters() if p.requires_grad)),
                  ("fusion_gate", tuple(p for p in self.model.weather_fusion.parameters()
                                        if p.requires_grad)))
        parameters = tuple(parameter for _, group in groups for parameter in group)
        cls = torch.autograd.grad(classification_loss, parameters,
                                  allow_unused=True, retain_graph=True)
        transfer = (torch.autograd.grad(weighted_transfer_loss, parameters,
                                       allow_unused=True, retain_graph=True)
                    if weighted_transfer_loss.requires_grad else (None,) * len(parameters))
        result = {}
        offset = 0
        for name, group in groups:
            count = sum(parameter.numel() for parameter in group)
            if not count:
                continue
            cls_sq = classification_loss.new_zeros(())
            transfer_sq = cls_sq.clone()
            dot = cls_sq.clone()
            for a, b in zip(cls[offset:offset + len(group)],
                            transfer[offset:offset + len(group)]):
                if a is not None:
                    cls_sq += a.detach().square().sum()
                if b is not None:
                    transfer_sq += b.detach().square().sum()
                if a is not None and b is not None:
                    dot += (a.detach() * b.detach()).sum()
            a, b, product = torch.stack((cls_sq.sqrt(), transfer_sq.sqrt(), dot)).cpu().tolist()
            result[name] = {
                "trainable_parameters": count,
                "classification_l2": a, "weighted_transfer_l2": b,
                "classification_rms": a / count**0.5,
                "weighted_transfer_rms": b / count**0.5,
                "transfer_to_classification": b / max(a, 1e-12),
                "cosine": product / (a * b) if a > 0 and b > 0 else None,
            }
            offset += len(group)
        self._write("backbone_gradients.jsonl", {
            "position": self.position, "sample_scope": "last training batch of reporting window",
            "gradients": result,
        })
        self.gradient_history.append(self.position)

    def report_training(self, position):
        self.position = position
        values = (self.loss_sums / self.steps).cpu().tolist()
        weights = self.model.weather_fusion.snapshot()
        self.weight_history[position] = weights
        record = {
            "position": position, "optimizer_updates": self.steps,
            "source_sample_exposures": self.source_seen,
            "target_sample_exposures": self.target_seen,
            "mean_classification_loss": values[0],
            "mean_transfer_loss": values[1], "mean_weighted_transfer_loss": values[2],
            "feature_norms": self.feature_summary(("source", "target")),
            "fusion_weights": weights,
            "optimizer_groups": self.optimizer_groups(),
        }
        self._write("training.jsonl", record)
        self._write("fusion_weights.jsonl", {"position": position, **weights})
        print("UDA_TRAINING_DIAGNOSTICS: " + json.dumps(record))
        self.loss_sums = None
        self.steps = self.source_seen = self.target_seen = 0
        self.domain = None

    def _capture_predictions(self, _module, _inputs, output):
        if self.partition is not None:
            self.probabilities.append(output.detach().softmax(dim=1).cpu())

    def begin_evaluation(self, partition, position):
        self.position = position
        self.partition = partition
        self.domain = partition + "_eval"
        self.probabilities = []
        self.evaluation_started = time.perf_counter()

    def finish_evaluation(self, metrics, dataset, views=1):
        labels = [int(label) for _, label in dataset.samples]
        count = len(labels)
        probabilities = torch.cat(self.probabilities).reshape(views, count, -1)
        combined = probabilities[0].clone()
        for view in probabilities[1:]:
            combined += view
        predicted = combined.argmax(1).tolist()
        combined /= views
        fused = metrics.get("fused", metrics)
        measured = classification_metrics(labels, predicted, len(self.class_ids))
        if any(abs(measured[name] - fused[name]) > 1e-10 for name in (
                "accuracy", "precision_macro", "recall_macro", "f1_macro")):
            raise RuntimeError("Diagnostic predictions differ from the reported evaluation")
        ce = F.nll_loss(combined.clamp_min(1e-12).log(), torch.tensor(labels),
                        reduction="none").tolist()
        partition = self.partition
        with (self.directory / f"predictions_{partition}.jsonl").open("a", encoding="utf-8") as stream:
            for index, (truth, pred, row, loss) in enumerate(zip(labels, predicted, combined.tolist(), ce)):
                pairs = getattr(dataset, "paired_samples", None)
                paths = pairs[index][:2] if pairs is not None else (dataset.samples[index][0], None)
                stream.write(json.dumps({
                    "method": self.method, "weather": self.args.target, "seed": self.args.seed,
                    "position": self.position, "partition": partition, "sample_index": index,
                    "sample_id": str(paths[0]), "vis_path": str(paths[0]),
                    "ir_path": str(paths[1]) if paths[1] is not None else None,
                    "truth_compact": truth, "truth_original": self.class_ids[truth],
                    "prediction_compact": pred, "prediction_original": self.class_ids[pred],
                    "confidence": max(row), "probabilities_compact_order": row,
                    "cross_entropy": loss, "correct": truth == pred,
                }, ensure_ascii=False) + "\n")
        recall = []
        class_loss = []
        matrix = measured["confusion_matrix"]
        for index, entries in enumerate(matrix):
            support = sum(entries)
            recall.append(entries[index] / support if support else None)
            losses = [loss for truth, loss in zip(labels, ce) if truth == index]
            class_loss.append(sum(losses) / len(losses) if losses else None)
        record = {
            "position": self.position, "partition": partition, "views": views,
            "original_class_ids": self.class_ids, "metrics": measured,
            "recall_by_class": recall, "cross_entropy_by_class": class_loss,
            "feature_norms": self.feature_summary((partition + "_eval",)),
            "fusion_weights": self.model.weather_fusion.snapshot(),
        }
        self._write("evaluation.jsonl", record)
        if partition == "source_train":
            self.source_history[self.position] = measured
            self.source_seconds += time.perf_counter() - self.evaluation_started
            print("UDA_SOURCE_EVALUATION: " + json.dumps(record))
        elif partition == "target_val":
            self.target_history[self.position] = measured
        self.partition = self.domain = None
        self.probabilities = []

    def evaluate_source(self, position):
        self.begin_evaluation("source_train", position)
        labels, predictions = [], []
        with evaluation_state(self.model, self.device), torch.no_grad():
            loader = DataLoader(
                self.source, batch_size=getattr(self.args, "batch_size", 16),
                shuffle=False, num_workers=0,
                generator=torch.Generator().manual_seed(0),
            )
            for images, truth in loader:
                outputs = self.forward(self.model, images.to(self.device))
                labels.extend(truth.tolist())
                predictions.extend(outputs.argmax(1).cpu().tolist())
        metrics = classification_metrics(labels, predictions, len(self.class_ids))
        self.finish_evaluation(metrics, self.source)

    def evaluate_target_train(self, position):
        if self.target_train is None:
            raise RuntimeError("Detailed diagnostics require the target training dataset")
        self.begin_evaluation("target_train", position)
        started = time.perf_counter()
        with evaluation_state(self.model, self.device), torch.no_grad():
            loader = DataLoader(
                ImageOnlyView(self.target_train), batch_size=getattr(self.args, "batch_size", 16),
                shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(0),
            )
            for images in loader:
                self.forward(self.model, images.to(self.device))
        probabilities = torch.cat(self.probabilities)
        if len(probabilities) != len(self.target_train):
            raise RuntimeError("Target prediction diagnostics did not cover every training pair")
        predictions = probabilities.argmax(1)
        record = {
            "position": position, "partition": "target_train", "labels_used": False,
            "samples": len(probabilities),
            "mean_confidence": float(probabilities.max(1).values.mean()),
            "mean_entropy": float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1).mean()),
            "original_class_ids": self.class_ids,
            "predicted_counts_compact": predictions.bincount(minlength=len(self.class_ids)).tolist(),
            "feature_norms": self.feature_summary(("target_train_eval",)),
            "fusion_weights": self.model.weather_fusion.snapshot(),
        }
        pairs = getattr(self.target_train, "paired_samples", None)
        with (self.directory / "predictions_target_train.jsonl").open("a", encoding="utf-8") as stream:
            for index, row in enumerate(probabilities.tolist()):
                paths = pairs[index][:2] if pairs is not None else (self.target_train.samples[index][0], None)
                prediction = int(predictions[index])
                stream.write(json.dumps({
                    "position": position, "partition": "target_train", "labels_used": False,
                    "sample_id": str(paths[0]), "vis_path": str(paths[0]),
                    "ir_path": str(paths[1]) if paths[1] is not None else None,
                    "prediction_compact": prediction, "prediction_original": self.class_ids[prediction],
                    "confidence": max(row), "probabilities_compact_order": row,
                }, ensure_ascii=False) + "\n")
        self._write("target_train_predictions.jsonl", record)
        self.target_prediction_history[position] = record
        self.target_prediction_seconds += time.perf_counter() - started
        self.partition = self.domain = None
        self.probabilities = []

    def validate_outputs(self, selected_position):
        expected = set(self.target_history)
        if not expected or selected_position not in expected:
            raise RuntimeError("Selected model has no recorded target evaluation")
        required = {
            "complete source evaluation": set(self.source_history),
            "full unlabelled target predictions": set(self.target_prediction_history),
            "weather weights": set(self.weight_history),
            "backbone gradient diagnostics": set(self.gradient_history),
        }
        errors = [f"{name} missing positions {sorted(expected - positions)}"
                  for name, positions in required.items() if not expected <= positions]
        event_files = {
            "training.jsonl": set(self.weight_history),
            "fusion_weights.jsonl": set(self.weight_history),
            "backbone_gradients.jsonl": set(self.weight_history),
            "evaluation.jsonl": expected,
            "target_train_predictions.jsonl": expected,
        }
        for filename, positions in event_files.items():
            path = self.directory / filename
            if not path.is_file():
                errors.append(f"missing {filename}")
                continue
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            recorded_positions = {record["position"] for record in records}
            if not positions <= recorded_positions:
                errors.append(f"{filename} incomplete reporting positions")
            if filename == "training.jsonl":
                for record in records:
                    if record["optimizer_updates"] <= 0 or record["source_sample_exposures"] <= 0:
                        errors.append(f"{filename} empty training window")
                    for domain in ("source", "target"):
                        if record[f"{domain}_sample_exposures"] > 0:
                            if not {"VIS", "IR"} <= set(record["feature_norms"].get(domain, {})):
                                errors.append(f"{filename} missing {domain} modality features")
            elif filename == "backbone_gradients.jsonl":
                for record in records:
                    if not {"VIS", "IR"} <= set(record["gradients"]):
                        errors.append(f"{filename} missing modality gradient")
        if not (self.directory / "run_configuration.jsonl").is_file():
            errors.append("missing run_configuration.jsonl")
        samples_checked = {}
        for partition, dataset in (("source_train", self.source), ("target_val", self.target_val),
                                   ("target_train", self.target_train)):
            path = self.directory / f"predictions_{partition}.jsonl"
            identities = {}
            if not path.is_file():
                errors.append(f"missing {path.name}")
                continue
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    position = record["position"]
                    seen = identities.setdefault(position, set())
                    if record["sample_id"] in seen:
                        errors.append(f"{partition} duplicate sample at {position}")
                    seen.add(record["sample_id"])
                    probabilities = record["probabilities_compact_order"]
                    if (len(probabilities) != len(self.class_ids) or
                            any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities) or
                            abs(sum(probabilities) - 1) > 1e-4):
                        errors.append(f"{partition} invalid probabilities at {position}")
                    if record["prediction_original"] != self.class_ids[record["prediction_compact"]]:
                        errors.append(f"{partition} label mapping mismatch at {position}")
                    if partition != "target_train":
                        if record["truth_original"] != self.class_ids[record["truth_compact"]]:
                            errors.append(f"{partition} ground-truth mapping mismatch at {position}")
                    elif record.get("labels_used") is not False:
                        errors.append(f"{partition} training labels exposed")
            for position in expected:
                if dataset is None or len(identities.get(position, ())) != len(dataset):
                    errors.append(f"{partition} sample coverage mismatch at {position}")
            samples_checked[partition] = sum(map(len, identities.values()))
        for position in expected:
            weights = self.weight_history.get(position, {})
            if not weights or abs(weights["vis"] + weights["ir"] - 1) > 1e-6:
                errors.append(f"invalid weather weights at {position}")
        report = {
            "status": "FAIL" if errors else "PASS", "selected_position": selected_position,
            "evaluation_positions": sorted(expected), "sample_rows_checked": samples_checked,
            "checks": list(required), "errors": errors,
        }
        (self.directory / "completeness.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("UDA_DIAGNOSTIC_COMPLETENESS: " + json.dumps(report))
        if errors:
            raise RuntimeError("Diagnostic completeness audit failed: " + "; ".join(errors))
        return report

    def summary(self, selected_position):
        self.validate_outputs(selected_position)
        weights = self.weight_history[selected_position]
        return {
            "fusion_weight_scope": "one global pair per method and target weather",
            "fusion_vis_weight": weights["vis"], "fusion_ir_weight": weights["ir"],
            "fusion_weights_learnable": weights["learnable"],
            "fusion_weight_minimum": weights["minimum"],
            "fusion_weather": self.args.target,
            "fusion_vis_weight_last": self.model.weather_fusion.snapshot()["vis"],
            "source_selected_accuracy": self.source_history[selected_position]["accuracy"],
            "diagnostic_source_eval_total_s": self.source_seconds,
            "diagnostic_target_eval_total_s": self.target_prediction_seconds,
            "diagnostics_complete": True,
            "diagnostics_dir": str(self.directory),
        }
