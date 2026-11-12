"""Three-modal adapters and DA objectives for the explicitly supervised VDS split.

The image/AIS interface is shared between methods. These are SDA extensions of
the named objectives, not claims to reproduce their original UDA benchmarks.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Transfer-Learning-Library"))
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy
from tllib.modules.grl import GradientReverseFunction
from tllib.self_training.mcc import MinimumClassConfusionLoss
from DeepCORAL.models import CORAL
from vds2raw.dataset import (
    CLASS_TO_ID, TRAIN_PARTITIONS, load_or_compute_statistics, partition_files,
)

METHODS = ("dann", "deepcoral", "cdan", "mdd", "mcc")
DEFAULTS = {
    "dann": {"lr": 0.0003, "trade_off": 0.35},
    "deepcoral": {"lr": 0.0005, "trade_off": 0.30},
    "cdan": {"lr": 0.0005, "trade_off": 1.0},
    "mdd": {"lr": 0.0025, "trade_off": 0.30},
    "mcc": {"lr": 0.002, "trade_off": 0.20},
}


def prepare_statistics(root, path):
    """Fit images and valid length/width only on the two training partitions.

    NPZ length/width already use the published training-only z scores. Fitting
    them again is harmless (approximately identity) and makes provenance local
    to this experiment. Heading sin/cos and navigation one-hots stay unscaled.
    """
    schema = json.loads((Path(root) / "feature_schema.json").read_text(encoding="utf-8"))
    if tuple(schema["train_partitions"]) != TRAIN_PARTITIONS or schema["feature_dim"] != 13:
        raise ValueError("AIS schema must use 13 fields and training-only scalers")
    images = load_or_compute_statistics(root, path)
    values = []
    masks = []
    for partition in TRAIN_PARTITIONS:
        for sample in partition_files(root, partition):
            with np.load(sample, allow_pickle=False) as item:
                values.append(item["ais"].astype(np.float64))
                masks.append(item["ais_mask"].astype(bool))
    values, masks = np.stack(values), np.stack(masks)
    mean, std = np.zeros(13), np.ones(13)
    for index in (2, 3):
        valid = values[masks[:, index], index]
        if not len(valid):
            raise ValueError(f"No valid training AIS values for feature {index}")
        mean[index] = valid.mean()
        std[index] = max(float(valid.std()), 1e-6)
    ais = {
        "mean": mean.tolist(), "std": std.tolist(),
        "standardized_indices": [2, 3], "feature_names": schema["ais_feature_names"],
        "fit_partitions": list(TRAIN_PARTITIONS), "fit_samples": len(values),
        "valid_counts": masks.sum(0).tolist(), "input_dimension": 13,
        "mask_dimension": 13,
    }
    payload = {**images.to_dict(), "ais_statistics": ais}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return images, ais


class AdaptationDataset(Dataset):
    """Full 13-field AIS view; does not change existing MCFT/TIR loaders."""

    def __init__(self, root, partition, images, ais, augment=False):
        self.paths = partition_files(root, partition)
        self.partition, self.augment = partition, augment
        if augment and partition not in TRAIN_PARTITIONS:
            raise ValueError("Evaluation partitions cannot use augmentation")
        self.rgb_mean = torch.tensor(images.rgb_mean).view(3, 1, 1)
        self.rgb_std = torch.tensor(images.rgb_std).view(3, 1, 1)
        self.nir_mean = torch.tensor(images.nir_mean).view(1, 1, 1)
        self.nir_std = torch.tensor(images.nir_std).view(1, 1, 1)
        self.ais_mean, self.ais_std = torch.tensor(ais["mean"]), torch.tensor(ais["std"])
        self.samples = [(str(p), CLASS_TO_ID[p.parent.name]) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as item:
            rgb = torch.from_numpy(item["rgb"].astype(np.float32))
            nir = torch.from_numpy(item["nir"].astype(np.float32))
            ais = torch.from_numpy(item["ais"].astype(np.float32))
            mask = torch.from_numpy(item["ais_mask"].astype(np.float32))
            label = int(item["label"])
        if label != CLASS_TO_ID[path.parent.name] or ais.shape != (13,):
            raise ValueError(f"Label/feature schema mismatch: {path}")
        if self.augment:
            # Preserve spatial registration and AIS/image heading consistency.
            if bool(torch.rand(()) < 0.5):
                rgb, nir, ais[0] = rgb.flip(-1), nir.flip(-1), -ais[0]
            if bool(torch.rand(()) < 0.5):
                rgb, nir, ais[1] = rgb.flip(-2), nir.flip(-2), -ais[1]
            top, left = torch.randint(0, 9, (2,)).tolist()
            rgb = F.pad(rgb, (4, 4, 4, 4), mode="reflect")[:, top:top+96, left:left+96]
            nir = F.pad(nir, (4, 4, 4, 4), mode="reflect")[:, top:top+96, left:left+96]
            rgb = rgb * (0.95 + 0.10 * torch.rand(()))
        return {
            "rgb": (rgb-self.rgb_mean)/self.rgb_std,
            "nir": (nir-self.nir_mean)/self.nir_std,
            "ais": ((ais-self.ais_mean)/self.ais_std)*mask,
            "ais_mask": mask, "label": label, "path": str(path),
        }


class PairedEpochSampler(Sampler):
    """Fixed full batches, complete coverage, explicit shorter-domain cycling."""

    def __init__(self, dataset, batch_size, steps, seed):
        self.size, self.batch_size, self.steps = len(dataset), batch_size, steps
        self.seed, self.epoch = int(seed), 0
        if batch_size < 2 or self.size < 2 or steps*batch_size < self.size:
            raise ValueError("Paired sampler must cover every item with batch >= 2")

    def __len__(self):
        return self.steps

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        indices = []
        while len(indices) < self.steps*self.batch_size:
            indices.extend(torch.randperm(self.size, generator=generator).tolist())
        for start in range(0, self.steps*self.batch_size, self.batch_size):
            yield indices[start:start+self.batch_size]


def reverse(value, coefficient):
    return GradientReverseFunction.apply(value, float(coefficient))


def cdan_entropy_loss(features, logits, discriminator, source_count, coefficient):
    """Device-independent CDAN+E formula from CDAN/pytorch/loss.py.

    The feature-probability outer product detaches probabilities. Entropy uses
    its own reversed gradient; both per-domain normalization denominators are
    detached, exactly as in the released formulation.
    """
    probabilities = logits.softmax(1)
    joint = torch.bmm(probabilities.detach().unsqueeze(2), features.unsqueeze(1)).flatten(1)
    domain_logits = discriminator(reverse(joint, coefficient)).flatten()
    truth = torch.cat((torch.ones(source_count, device=features.device),
                       torch.zeros(len(features)-source_count, device=features.device)))
    entropy = -(probabilities * (probabilities+1e-5).log()).sum(1)
    weights = 1 + (-reverse(entropy, coefficient)).exp()
    source_weights, target_weights = weights[:source_count], weights[source_count:]
    weights = torch.cat((source_weights/source_weights.sum().detach(),
                         target_weights/target_weights.sum().detach()))
    loss = (F.binary_cross_entropy_with_logits(domain_logits, truth, reduction="none")
            * weights).sum()/weights.sum().detach()
    return loss, float(((domain_logits >= 0) == truth.bool()).float().mean().detach())


class AdaptationModel(nn.Module):
    def __init__(self, method, weights_path=None, pretrained=True):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.rgb_encoder = resnet18(weights=None)
        if pretrained:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing ResNet-18 weights: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not digest.startswith("f37072fd"):
                raise ValueError(f"ResNet-18 checkpoint checksum mismatch: {digest}")
            self.rgb_encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.rgb_encoder.fc = nn.Identity()
        self.nir_encoder = deepcopy(self.rgb_encoder)
        conv = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(self.rgb_encoder.conv1.weight.mean(1, keepdim=True))
        self.nir_encoder.conv1 = conv
        self.ais_encoder = nn.Sequential(nn.Linear(26, 128), nn.ReLU(),
                                         nn.Dropout(0.1), nn.Linear(128, 128), nn.ReLU())
        self.modality_logits = nn.Parameter(torch.zeros(3))
        if method == "deepcoral":
            self.bottleneck = nn.Identity()
            dimension = 1152
        else:
            self.bottleneck = nn.Sequential(nn.Linear(1152, 256), nn.BatchNorm1d(256), nn.ReLU())
            dimension = 256
        self.head = nn.Linear(dimension, 3)
        nn.init.normal_(self.head.weight, 0, 0.005)
        if method in {"dann", "cdan"}:
            self.discriminator = nn.Sequential(
                nn.Linear(dimension*(3 if method == "cdan" else 1), 512),
                nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 512), nn.ReLU(),
                nn.Dropout(0.5), nn.Linear(512, 1),
            )
        if method == "mdd":
            self.adv_head = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(),
                                           nn.Dropout(0.5), nn.Linear(256, 3))
        self.mdd = ClassificationMarginDisparityDiscrepancy(margin=4)
        self.mcc = MinimumClassConfusionLoss(temperature=2.5)
        self.last_features = None

    def modality_weights(self):
        return 0.05 + 0.85*self.modality_logits.softmax(0)

    def encode(self, batch):
        # The validity mask is an explicit part of this 13-field experiment.
        branches = (self.rgb_encoder(batch["rgb"]), self.nir_encoder(batch["nir"]),
                    self.ais_encoder(torch.cat((batch["ais"]*batch["ais_mask"], batch["ais_mask"]), 1)))
        self.last_features = tuple(branch.detach() for branch in branches)
        weights = self.modality_weights()
        return self.bottleneck(torch.cat([3*weights[i]*v for i, v in enumerate(branches)], 1))

    def forward(self, batch):
        return self.head(self.encode(batch))

    def objective(self, source, target, update, planned_updates, args):
        source_count = len(source["label"])
        inputs = {key: torch.cat((source[key], target[key]), 0)
                  for key in ("rgb", "nir", "ais", "ais_mask")}
        features = self.encode(inputs)
        logits = self.head(features)
        ys, yt = logits[:source_count], logits[source_count:]
        ce_source = F.cross_entropy(ys, source["label"], label_smoothing=args.label_smoothing)
        ce_target = F.cross_entropy(yt, target["label"], label_smoothing=args.label_smoothing)
        classification = (args.source_weight*ce_source + args.target_weight*ce_target)/(args.source_weight+args.target_weight)
        progress = update/max(planned_updates, 1)
        coefficient = 2/(1+math.exp(-10*progress))-1
        domain_accuracy = None
        if self.method == "dann":
            coefficient = min(coefficient, 0.45)
            predictions = self.discriminator(reverse(features, coefficient)).flatten()
            domains = torch.cat((torch.ones(source_count, device=logits.device),
                                 torch.zeros(len(yt), device=logits.device)))
            transfer = F.binary_cross_entropy_with_logits(predictions, domains)
            domain_accuracy = float(((predictions >= 0) == domains.bool()).float().mean().detach())
        elif self.method == "deepcoral":
            coefficient = 0.0
            transfer = CORAL(ys, yt)
        elif self.method == "cdan":
            transfer, domain_accuracy = cdan_entropy_loss(features, logits, self.discriminator, source_count, coefficient)
        elif self.method == "mdd":
            # Preserve the official MDD GRL schedule and negative discrepancy.
            coefficient = 0.2/(1+math.exp(-update/1000))-0.1
            adversarial = self.adv_head(reverse(features, coefficient))
            transfer = -self.mdd(ys, adversarial[:source_count], yt, adversarial[source_count:])
        else:
            coefficient = 0.0
            transfer = self.mcc(yt)
        warmup = args.adaptation_warmup_epochs*args.steps_per_epoch
        ramp = max(args.adaptation_ramp_epochs*args.steps_per_epoch, 1)
        weight = args.trade_off*min(max((update+1-warmup)/ramp, 0), 1)
        return classification+weight*transfer, {
            "classification": classification, "source_ce": ce_source, "target_ce": ce_target,
            "transfer": transfer, "weighted_transfer": weight*transfer,
            "transfer_weight": weight, "grl_coefficient": coefficient,
            "domain_accuracy": domain_accuracy, "source_logits": ys, "target_logits": yt,
        }
