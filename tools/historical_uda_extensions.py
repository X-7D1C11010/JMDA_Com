"""Optional, separately reported experiments for the pinned VIS+IR baselines."""

from copy import copy, deepcopy
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from data_loader_options import data_loader_kwargs


class ContinuousShuffleSampler(Sampler):
    """Exhaust each permutation before starting the next, even across batches."""

    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("continuous source sampling requires a nonempty dataset")
        self.size = size
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.size, generator=generator).tolist()


class ImageOnlyView(Dataset):
    """Expose images from the target training partition for BN calibration."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0]


class TargetBNProbe:
    """Compare raw and target-calibrated predictions at identical model weights."""

    def __init__(self, model, target_training_dataset, transform, forward,
                 device, batch_size):
        from uda_multimodal import ir_transform_from_vis

        if batch_size < 2:
            raise ValueError("Target BN calibration batch size must be at least two")
        dataset = copy(target_training_dataset)
        dataset.transform = transform
        dataset.ir_transform = ir_transform_from_vis(transform)
        if len(dataset) < 2:
            raise ValueError("Target BN calibration requires at least two training pairs")
        indices = list(range(len(dataset)))
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        if len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        self.loader = DataLoader(
            ImageOnlyView(dataset), batch_sampler=batches, num_workers=0,
            generator=torch.Generator().manual_seed(0),
        )
        self.model = model
        self.forward = forward
        self.device = device
        self.best_raw = None
        self.best_raw_position = None
        self.calibration_seconds = 0.0
        self.fit_samples = len(dataset)

    def evaluate(self, raw_metrics, evaluate, position):
        fused_raw = raw_metrics.get("fused", raw_metrics)
        if (self.best_raw is None or
                (fused_raw["accuracy"], fused_raw["f1_macro"]) >
                (self.best_raw["accuracy"], self.best_raw["f1_macro"])):
            self.best_raw = dict(fused_raw)
            self.best_raw_position = position
        modes = {module: module.training for module in self.model.modules()}
        norms = [module for module in self.model.modules()
                 if isinstance(module, nn.modules.batchnorm._BatchNorm)
                 and module.track_running_stats]
        state = {module: (
            module.running_mean.clone(), module.running_var.clone(),
            module.num_batches_tracked.clone(), module.momentum,
        ) for module in norms}
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cuda_devices = ([self.device.index if self.device.index is not None
                         else torch.cuda.current_device()]
                        if self.device.type == "cuda" else [])
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                self.model.eval()
                for norm in norms:
                    norm.reset_running_stats()
                    norm.train()
                if cuda_devices:
                    torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                seen = 0
                with torch.no_grad():
                    for images in self.loader:
                        count = len(images)
                        for norm in norms:
                            norm.momentum = count / (seen + count)
                        self.forward(self.model, images.to(self.device))
                        seen += count
                if cuda_devices:
                    torch.cuda.synchronize(self.device)
                self.calibration_seconds += time.perf_counter() - started
                self.model.eval()
                with torch.no_grad():
                    calibrated = evaluate()
        finally:
            for norm, (running_mean, running_var, batches_seen, momentum) in state.items():
                norm.running_mean.copy_(running_mean)
                norm.running_var.copy_(running_var)
                norm.num_batches_tracked.copy_(batches_seen)
                norm.momentum = momentum
            for module, training in modes.items():
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        fused_calibrated = calibrated.get("fused", calibrated)
        print("UDA_TARGET_BN_COMPARISON: " + json.dumps({
            "position": position,
            "fit_partition": "target_train",
            "fit_pairs": self.fit_samples,
            "raw": fused_raw,
            "target_bn": fused_calibrated,
        }))
        return calibrated

    def summary(self):
        return {
            "prediction_protocol": "target_train_BN_calibrated",
            "bn_fit_partition": "target_train",
            "bn_fit_pairs": self.fit_samples,
            "bn_calibration_total_s": round(self.calibration_seconds, 4),
            "raw_best_acc": self.best_raw["accuracy"],
            "raw_best_position": self.best_raw_position,
            "raw_best_precision_macro": self.best_raw["precision_macro"],
            "raw_best_recall_macro": self.best_raw["recall_macro"],
            "raw_best_f1_macro": self.best_raw["f1_macro"],
        }


class VisibleJitter(transforms.ColorJitter):
    """Identify the VIS photometric operation when building the IR transform."""


def center_view_transform(original):
    normalization = next(
        operation for operation in original.transforms
        if isinstance(operation, transforms.Normalize)
    )
    return transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(), deepcopy(normalization),
    ])


def full_frame_training_transform(original):
    normalization = next(
        operation for operation in original.transforms
        if isinstance(operation, transforms.Normalize)
    )
    print("Augmentation variant: full-frame 224x224, synchronized HFlip, "
          "VIS brightness/contrast=0.15, IR contrast=0.10")
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        VisibleJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalization,
    ])


def pretrain_ir_encoder(encoder, source_dataset, num_classes, args, device):
    """Fit only the IR encoder and a disposable head on labelled sunny pairs."""
    started = time.monotonic()
    rng_python, rng_numpy = random.getstate(), np.random.get_state()
    cuda_devices = [device.index if device.index is not None else
                    torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            head = nn.Linear(encoder.out_features, num_classes).to(device)
            optimizer = torch.optim.AdamW(
                [*encoder.parameters(), *head.parameters()],
                lr=2e-4, weight_decay=5e-4,
            )
            loader = DataLoader(
                source_dataset, batch_size=args.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(args.seed),
                **data_loader_kwargs(args),
            )
            encoder.train()
            print("IR pretrain: source_train only, epochs=10, AdamW lr=0.0002")
            for epoch in range(1, 11):
                loss_sum = count = 0
                for pairs, labels in loader:
                    labels = labels.to(device)
                    ir = pairs[:, 3:].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head(encoder(ir)), labels)
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(labels)
                    count += len(labels)
                print(f"IR pretrain [{epoch}/10] source_n={count} CE={loss_sum / count:.4f}")
    finally:
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
    print(f"IR pretrain seconds={time.monotonic() - started:.1f}")


class EMATargetTeacher:
    """Distill confident predictions on an alternate, paired target view."""

    def __init__(self, student, predict):
        self.model = deepcopy(student).eval().requires_grad_(False)
        self.predict = predict
        self.accepted = next(student.parameters()).new_zeros((), dtype=torch.long)
        self.seen = 0
        print("EMA extension: decay=0.99, confidence=0.90, loss_weight=0.10, "
              "teacher view=paired horizontal flip, evaluation=student")

    def loss(self, student_logits, target_pairs):
        with torch.no_grad():
            teacher_probs = self.predict(self.model, target_pairs.flip(-1)).softmax(dim=1)
            confident = teacher_probs.amax(dim=1).ge(0.90)
            self.accepted += confident.sum()
            self.seen += len(confident)
        per_sample = F.kl_div(
            student_logits.log_softmax(dim=1), teacher_probs, reduction="none"
        ).sum(dim=1)
        return 0.10 * (per_sample * confident).sum() / confident.sum().clamp_min(1)

    @torch.no_grad()
    def update(self, student):
        for average, current in zip(self.model.parameters(), student.parameters()):
            average.lerp_(current.detach(), 0.01)
        for average, current in zip(self.model.buffers(), student.buffers()):
            average.copy_(current)

    def report(self):
        print(f"EMA target pairs accepted={self.accepted.item()}/{self.seen}")
