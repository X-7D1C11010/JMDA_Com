"""Optional, separately reported experiments for the pinned VIS+IR baselines."""

from copy import deepcopy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
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


class VisibleJitter(transforms.ColorJitter):
    """Identify the VIS photometric operation when building the IR transform."""


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
