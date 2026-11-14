"""Three-modal VDS adapters with source-only supervised UDA objectives."""

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
from torchvision.models import resnet18, convnext_small

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Transfer-Learning-Library"))
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy
from tllib.modules.grl import GradientReverseFunction
from tllib.self_training.mcc import MinimumClassConfusionLoss
from DeepCORAL.models import CORAL
from vds2raw.dataset import (
    CLASS_TO_ID, TRAIN_PARTITIONS, load_or_compute_statistics, partition_files,
    SAFE_AIS_FEATURE_NAMES,
)

METHODS = ("dann", "deepcoral", "cdan", "mdd", "mcc")
DEFAULTS = {
    "dann": {"lr": 0.0003, "trade_off": 0.35},
    "deepcoral": {"lr": 0.0005, "trade_off": 0.30},
    "cdan": {"lr": 0.0005, "trade_off": 1.0},
    "mdd": {"lr": 0.0025, "trade_off": 0.30},
    "mcc": {"lr": 0.002, "trade_off": 0.20},
}


def resolve_training_settings(args):
    settings = {**DEFAULTS[args.method], "epochs": 100, "ais_lr_multiplier": 1.0,
                "val_plateau_patience": 0, "val_plateau_factor": 0.5, "val_plateau_min_scale": 0.1,
                "early_stopping_patience": 0, "early_stopping_monitor": "accuracy_f1",
                "early_stopping_min_epochs": 0, "validation_loss_min_delta": 0.0001,
                "source_class_balance": False, "domain_forward": "reference", "log_feature_moments": False,
                "adaptation_warmup_epochs": 0, "feature_normalization": "none",
                "validation_control_start_epoch": 1, "modality_diagnostics": False, "label_smoothing": 0.}
    if args.profile in {"mcc_v2", "mcc_v3"}:
        if args.method != "mcc":
            raise ValueError(f"{args.profile} is reserved for MCC")
        settings.update(trade_off=0.50, ais_lr_multiplier=5.0,
                        adaptation_warmup_epochs=20, adaptation_ramp_epochs=20,
                        source_class_balance=True, log_feature_moments=True)
        if args.profile == "mcc_v3":
            settings.update(feature_normalization="l2", val_plateau_patience=12,
                            early_stopping_monitor="cross_entropy", early_stopping_patience=25,
                            early_stopping_min_epochs=60, validation_control_start_epoch=40,
                            modality_diagnostics=True)
    if args.profile == "cdan_safe_v4":
        if args.method != "cdan":
            raise ValueError("cdan_safe_v4 is reserved for CDAN")
        settings.update(lr=.0002, trade_off=.50, ais_lr_multiplier=5., feature_normalization="l2",
                        source_class_balance=True, adaptation_warmup_epochs=20, adaptation_ramp_epochs=20,
                        val_plateau_patience=12, validation_control_start_epoch=40,
                        early_stopping_monitor="cross_entropy", early_stopping_patience=25,
                        early_stopping_min_epochs=60, modality_diagnostics=True, log_feature_moments=True)
    if args.profile == "optimized" and args.method in {"mdd", "mcc"}:
        settings.update(ais_lr_multiplier=5.0)
        if args.method == "mdd":
            # 250 x 8 complete-domain updates = 2000 on the stated VDS split.
            # Keep the reference GRL clock; extend the actual optimization
            # budget instead of silently changing the GRL formula.
            settings.update(epochs=250, trade_off=0.60)
        else:
            settings.update(trade_off=0.50)
    if args.profile == "optimized" and args.method == "cdan":
        # The observed validation CE keeps improving while ACC/F1 stalls.
        # Control late fitting with CE without changing best-ACC selection.
        settings.update(val_plateau_patience=12, early_stopping_patience=20,
                        early_stopping_monitor="cross_entropy", early_stopping_min_epochs=50)
    for key, value in settings.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def source_classification_weights(samples, num_classes):
    """Inverse-frequency CE weights from source labels only; retain all samples."""
    counts = torch.bincount(torch.tensor([int(label) for _, label in samples]), minlength=num_classes)
    if counts.numel() != num_classes or bool((counts == 0).any()):
        raise ValueError("Every configured class must appear in the labelled source")
    return counts, counts.sum()/(num_classes*counts.float())


def domain_moments(values, source_count):
    source, target = values[:source_count].detach(), values[source_count:].detach()
    return {"source_mean_norm": float(source.mean(0).norm()), "target_mean_norm": float(target.mean(0).norm()),
            "mean_gap_L2": float((source.mean(0)-target.mean(0)).norm()),
            "std_gap_L2": float((source.std(0, unbiased=False)-target.std(0, unbiased=False)).norm())}


def ais_input_protocol(schema, fields):
    if fields not in {"heading_size", "full13"}:
        raise ValueError(f"Unknown AIS protocol: {fields}")
    names = schema["ais_feature_names"]
    if len(names) != 13 or tuple(names[:4]) != SAFE_AIS_FEATURE_NAMES:
        raise ValueError("AIS schema does not match heading_sin/cos, length_z, width_z")
    indices = list(range(4 if fields == "heading_size" else 13))
    return {"protocol": fields, "feature_indices": indices,
            "feature_names": [names[i] for i in indices],
            "excluded_features": [name for i, name in enumerate(names) if i not in indices],
            "input_dimension": len(indices), "mask_dimension": len(indices)}


def prepare_statistics(root, path, ais_fields="full13"):
    """Fit images and valid length/width only on the two training partitions.

    NPZ length/width already use the published training-only z scores. Fitting
    them again is harmless (approximately identity) and makes provenance local
    to this experiment. Heading sin/cos and navigation one-hots stay unscaled.
    """
    schema = json.loads((Path(root) / "feature_schema.json").read_text(encoding="utf-8"))
    if tuple(schema["train_partitions"]) != TRAIN_PARTITIONS or schema["feature_dim"] != 13:
        raise ValueError("AIS schema must use 13 fields and training-only scalers")
    protocol = ais_input_protocol(schema, ais_fields)
    indices = protocol["feature_indices"]
    images = load_or_compute_statistics(root, path)
    values = []
    masks = []
    for partition in TRAIN_PARTITIONS:
        for sample in partition_files(root, partition):
            with np.load(sample, allow_pickle=False) as item:
                values.append(item["ais"][indices].astype(np.float64))
                masks.append(item["ais_mask"][indices].astype(bool))
    values, masks = np.stack(values), np.stack(masks)
    mean, std = np.zeros(len(indices)), np.ones(len(indices))
    for index in (2, 3):
        valid = values[masks[:, index], index]
        if not len(valid):
            raise ValueError(f"No valid training AIS values for feature {index}")
        mean[index] = valid.mean()
        std[index] = max(float(valid.std()), 1e-6)
    ais = {
        "mean": mean.tolist(), "std": std.tolist(),
        "standardized_indices": [2, 3], **protocol,
        "fit_partitions": list(TRAIN_PARTITIONS), "fit_samples": len(values),
        "valid_counts": masks.sum(0).tolist(),
    }
    payload = {**images.to_dict(), "ais_statistics": ais}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return images, ais


class AdaptationDataset(Dataset):
    """Project AIS values AND masks before normalization or model input."""

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
        self.ais_indices = ais.get("feature_indices", list(range(len(ais["mean"]))))
        self.labelled = partition != "target_train"
        self.samples = [(str(p), CLASS_TO_ID[p.parent.name] if self.labelled else -1)
                        for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as item:
            rgb = torch.from_numpy(item["rgb"].astype(np.float32))
            nir = torch.from_numpy(item["nir"].astype(np.float32))
            if item["ais"].shape != (13,) or item["ais_mask"].shape != (13,):
                raise ValueError(f"Invalid stored AIS shape: {path}")
            ais = torch.from_numpy(item["ais"][self.ais_indices].astype(np.float32))
            mask = torch.from_numpy(item["ais_mask"][self.ais_indices].astype(np.float32))
            # Do not even access the target-training NPZ label field.
            label = int(item["label"]) if self.labelled else None
        if self.labelled and label != CLASS_TO_ID[path.parent.name]:
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
        result = {
            "rgb": (rgb-self.rgb_mean)/self.rgb_std,
            "nir": (nir-self.nir_mean)/self.nir_std,
            "ais": ((ais-self.ais_mean)/self.ais_std)*mask,
            "ais_mask": mask, "path": str(path),
        }
        if self.labelled:
            result["label"] = label
        return result


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


def cdan_entropy_loss(features, logits, discriminator, source_count, coefficient, feature_coefficient=None):
    """Device-independent CDAN+E formula from CDAN/pytorch/loss.py.

    The feature-probability outer product detaches probabilities. Entropy uses
    its own reversed gradient; both per-domain normalization denominators are
    detached, exactly as in the released formulation.
    """
    probabilities = logits.softmax(1)
    joint = torch.bmm(probabilities.detach().unsqueeze(2), features.unsqueeze(1)).flatten(1)
    domain_logits = discriminator(reverse(joint, coefficient if feature_coefficient is None else feature_coefficient)).flatten()
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
    def __init__(self, method, weights_path=None, pretrained=True, *, num_classes=3, use_ais=True, ir_channels=1,
                 feature_normalization="none", ais_dimensions=13, auxiliary_source_weight=0.,
                 image_backbone="resnet18", freeze_image_encoders=False, unit_method_features=False,
                 encoder_training_scope=None, classifier_layout="reference"):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        if num_classes < 2 or ir_channels not in (1, 3):
            raise ValueError("Invalid class count or infrared channel count")
        self.num_classes, self.use_ais = num_classes, bool(use_ais)
        if ais_dimensions not in (4, 13):
            raise ValueError("AIS must use either heading/size (4) or explicit historical full13")
        self.ais_dimensions = ais_dimensions
        if auxiliary_source_weight < 0:
            raise ValueError("Auxiliary source supervision weight cannot be negative")
        self.auxiliary_source_weight = float(auxiliary_source_weight)
        if feature_normalization not in {"none", "l2", "unit"}:
            raise ValueError("Unknown feature normalization")
        self.feature_normalization = feature_normalization
        if classifier_layout not in {"reference", "linear"}:
            raise ValueError("Unknown classifier layout")
        self.classifier_layout = classifier_layout
        self.eval_disabled_modalities = ()
        self.register_buffer("source_class_weights", None)
        self.feature_names = ("RGB", "NIR", "AIS") if use_ais else ("VIS", "IR")
        self.input_keys = ("rgb", "nir", "ais", "ais_mask") if use_ais else ("rgb", "nir")
        if image_backbone not in {"resnet18", "convnext_small"}:
            raise ValueError("Unknown image backbone")
        if image_backbone == "convnext_small" and (use_ais or ir_channels != 3):
            raise ValueError("ConvNeXt screening is defined for paired weather VIS/IR only")
        scope = encoder_training_scope or ("frozen" if freeze_image_encoders else "all")
        if scope not in {"all", "frozen", "ir_last_stage", "last_stages"}:
            raise ValueError("Unknown encoder training scope")
        if freeze_image_encoders and scope != "frozen":
            raise ValueError("Do not combine fully frozen encoders with a trainable encoder scope")
        if scope in {"ir_last_stage", "last_stages"} and image_backbone != "convnext_small":
            raise ValueError("Last-stage screening is defined for the ConvNeXt weather encoder")
        self.image_backbone, self.encoder_training_scope = image_backbone, scope
        self.freeze_image_encoders = scope == "frozen"
        self.unit_method_features = bool(unit_method_features)
        self.image_feature_dim = 512 if image_backbone == "resnet18" else 768
        self.rgb_encoder = resnet18(weights=None) if image_backbone == "resnet18" else convnext_small(weights=None)
        if pretrained:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {image_backbone} weights: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected_hash = "f37072fd" if image_backbone == "resnet18" else "0c510722"
            if not digest.startswith(expected_hash):
                raise ValueError(f"{image_backbone} checkpoint checksum mismatch: {digest}")
            self.rgb_encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        if image_backbone == "resnet18":
            self.rgb_encoder.fc = nn.Identity()
        else:
            self.rgb_encoder.classifier[-1] = nn.Identity()
        self.nir_encoder = deepcopy(self.rgb_encoder)
        if ir_channels == 1:
            conv = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv.weight.copy_(self.rgb_encoder.conv1.weight.mean(1, keepdim=True))
            self.nir_encoder.conv1 = conv
        if scope != "all":
            self.rgb_encoder.requires_grad_(False).eval()
            self.nir_encoder.requires_grad_(False).eval()
        if scope in {"ir_last_stage", "last_stages"}:
            encoders = (self.nir_encoder,) if scope == "ir_last_stage" else (self.rgb_encoder, self.nir_encoder)
            for encoder in encoders:
                encoder.features[-1].requires_grad_(True)
                encoder.classifier[0].requires_grad_(True)
        if use_ais:
            self.ais_encoder = nn.Sequential(nn.Linear(2*ais_dimensions, 128), nn.ReLU(),
                                             nn.Dropout(0.1), nn.Linear(128, 128), nn.ReLU())
        self.modality_logits = nn.Parameter(torch.zeros(3 if use_ais else 2))
        fused_dimension = 2*self.image_feature_dim+(128 if use_ais else 0)
        if method == "deepcoral" or classifier_layout == "linear":
            # Opt-in weather probe: retain the pretrained representation and
            # isolate adaptation from the learned reference bottleneck.
            self.bottleneck = nn.Identity()
            dimension = fused_dimension
        elif method == "cdan":
            # Released ResNetFc uses a linear bottleneck, without BN/ReLU.
            self.bottleneck = nn.Linear(fused_dimension, 256)
            nn.init.normal_(self.bottleneck.weight, 0, 0.005)
            nn.init.constant_(self.bottleneck.bias, 0.1)
            dimension = 256
        else:
            self.bottleneck = nn.Sequential(nn.Linear(fused_dimension, 256), nn.BatchNorm1d(256), nn.ReLU())
            if method == "mdd":
                self.bottleneck.append(nn.Dropout(0.5))
                nn.init.normal_(self.bottleneck[0].weight, 0, 0.005)
                nn.init.constant_(self.bottleneck[0].bias, 0.1)
            dimension = 256
        self.head = nn.Linear(dimension, num_classes)
        nn.init.normal_(self.head.weight, 0, 0.005)
        if method in {"dann", "cdan"}:
            hidden = 1024 if method == "cdan" else 512
            self.discriminator = nn.Sequential(
                nn.Linear(dimension*(num_classes if method == "cdan" else 1), hidden),
                nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Dropout(0.5), nn.Linear(hidden, 1),
            )
            if method == "cdan":
                for layer in (self.bottleneck, self.head, *self.discriminator):
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        nn.init.zeros_(layer.bias)
        if method == "mdd":
            if classifier_layout == "linear":
                self.adv_head = nn.Linear(dimension, num_classes)
            else:
                self.head = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(),
                                         nn.Dropout(0.5), nn.Linear(256, num_classes))
                self.adv_head = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(),
                                             nn.Dropout(0.5), nn.Linear(256, num_classes))
            for head in (self.head, self.adv_head):
                for layer in ((head,) if isinstance(head, nn.Linear) else (head[0], head[3])):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.zeros_(layer.bias)
        self.mdd = ClassificationMarginDisparityDiscrepancy(margin=4)
        self.mcc = MinimumClassConfusionLoss(temperature=2.5)
        if auxiliary_source_weight:
            self.auxiliary_heads = nn.ModuleList(nn.Linear(width, num_classes)
                                                for width in ((self.image_feature_dim, self.image_feature_dim, 128)
                                                              if use_ais else (self.image_feature_dim, self.image_feature_dim)))
        self.last_features = None

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "encoder_training_scope", "all") in {"frozen", "ir_last_stage", "last_stages"}:
            # Frozen ConvNeXt must also keep stochastic depth disabled. Heads,
            # discriminators and the feature bottleneck follow normal mode.
            self.rgb_encoder.eval()
            self.nir_encoder.eval()
        return self

    def modality_weights(self):
        return 0.05 + (1-0.05*self.modality_logits.numel())*self.modality_logits.softmax(0)

    def encode(self, batch):
        # Only the mask entries corresponding to retained AIS fields enter.
        branches = (self.rgb_encoder(batch["rgb"]), self.nir_encoder(batch["nir"]))
        if getattr(self, "use_ais", True):
            expected = getattr(self, "ais_dimensions", 13)
            if batch["ais"].shape[1:] != (expected,) or batch["ais_mask"].shape[1:] != (expected,):
                raise ValueError("AIS value/mask width differs from the declared model protocol")
            branches += (self.ais_encoder(torch.cat((batch["ais"]*batch["ais_mask"], batch["ais_mask"]), 1)),)
        self.last_features = tuple(branch.detach() for branch in branches)
        normalization = getattr(self, "feature_normalization", "none")
        if normalization in {"l2", "unit"}:
            # Give each modality the same norm before learned weighting,
            # without fitting any validation/test statistics.
            scale = math.sqrt(getattr(self, "image_feature_dim", 512)) if normalization == "l2" else 1.
            branches = tuple(F.normalize(v, dim=1, eps=1e-8)*scale for v in branches)
        self.last_normalized_features = tuple(branch.detach() for branch in branches)
        branches_before_ablation = branches
        disabled = getattr(self, "eval_disabled_modalities", ())
        if disabled:
            if self.training:
                raise RuntimeError("Feature ablation is restricted to evaluation")
            branches = tuple(torch.zeros_like(v) if i in disabled else v for i, v in enumerate(branches))
        self.last_branch_logits = (tuple(head(branch) for head, branch in zip(self.auxiliary_heads, branches))
                                   if hasattr(self, "auxiliary_heads") else ())
        weights = self.modality_weights()
        fused = torch.cat([len(branches)*weights[i]*v for i, v in enumerate(branches)], 1)
        if normalization == "unit":
            # Preserve the unmasked scale during feature-zeroing diagnostics.
            # Otherwise masking a branch would also amplify the other branch.
            unmasked = (torch.cat([len(branches)*weights[i]*v for i, v in enumerate(branches_before_ablation)], 1)
                        if disabled else fused)
            fused = fused/unmasked.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.last_fused = fused.detach()
        method_features = self.bottleneck(fused)
        return (F.normalize(method_features, dim=1, eps=1e-8)
                if getattr(self, "unit_method_features", False) else method_features)

    def forward(self, batch):
        return self.head(self.encode(batch))

    def set_bn_domain(self, domain):
        if getattr(self, "domain_running_stats", False):
            from weather.domain_running_stats import set_domain
            set_domain(self, domain)

    def objective(self, source, target, update, planned_updates, args):
        source_count = len(source["label"])
        if self.method == "mdd" and source_count != len(target["rgb"]):
            raise ValueError("The released MDD loss requires equal source/target batch sizes; use full paired batches")
        log_moments = getattr(args, "log_feature_moments", False)
        if (self.method in {"dann", "deepcoral", "cdan"} and getattr(args, "domain_forward", "reference") == "reference"
                or getattr(self, "domain_running_stats", False)):
            # Preserve the separate domain forwards in these released loops.
            self.set_bn_domain("source")
            fs = self.encode(source)
            source_fused = self.last_fused if log_moments else None
            source_branches = self.last_features
            source_normalized = self.last_normalized_features
            source_auxiliary = self.last_branch_logits
            ys = self.head(fs)
            self.set_bn_domain("target")
            ft = self.encode(target)
            if log_moments:
                fused_for_diagnostics = torch.cat((source_fused, self.last_fused))
            target_branches = self.last_features
            target_normalized = self.last_normalized_features
            target_auxiliary = self.last_branch_logits
            yt = self.head(ft)
            self.last_features = tuple(torch.cat((s, t)) for s, t in zip(source_branches, target_branches))
            self.last_normalized_features = (tuple(torch.cat((s, t)) for s, t in zip(source_normalized, target_normalized))
                                             if getattr(self, "feature_normalization", "none") in {"l2", "unit"} else self.last_features)
            self.last_branch_logits = tuple(torch.cat((s, t)) for s, t in zip(source_auxiliary, target_auxiliary))
            features, logits = torch.cat((fs, ft)), torch.cat((ys, yt))
        else:
            # MDD/MCC released examples concatenate the domains before BN.
            inputs = {key: torch.cat((source[key], target[key]), 0)
                      for key in getattr(self, "input_keys", ("rgb", "nir", "ais", "ais_mask"))}
            features = self.encode(inputs)
            if log_moments:
                fused_for_diagnostics = self.last_fused
            logits = self.head(features)
        ys, yt = logits[:source_count], logits[source_count:]
        ce_source = F.cross_entropy(ys, source["label"], weight=getattr(self, "source_class_weights", None),
                                   label_smoothing=args.label_smoothing)
        classification = ce_source
        auxiliary = tuple(F.cross_entropy(branch[:source_count], source["label"],
                                          weight=getattr(self, "source_class_weights", None),
                                          label_smoothing=args.label_smoothing)
                          for branch in self.last_branch_logits)
        auxiliary_ce = torch.stack(auxiliary).mean() if auxiliary else ce_source.new_zeros(())
        classification = classification + getattr(self, "auxiliary_source_weight", 0.)*auxiliary_ce
        progress = update/max(planned_updates, 1)
        coefficient = 2/(1+math.exp(-10*progress))-1
        domain_accuracy = None
        if self.method == "dann":
            predictions = self.discriminator(reverse(features, coefficient)).flatten()
            domains = torch.cat((torch.ones(source_count, device=logits.device),
                                 torch.zeros(len(yt), device=logits.device)))
            transfer = (F.binary_cross_entropy_with_logits(predictions[:source_count], domains[:source_count])
                        + F.binary_cross_entropy_with_logits(predictions[source_count:], domains[source_count:]))
            domain_accuracy = float(((predictions >= 0) == domains.bool()).float().mean().detach())
        elif self.method == "deepcoral":
            coefficient = 0.0
            transfer = CORAL(ys, yt)
        elif self.method == "cdan":
            entropy_coefficient = 2/(1+math.exp(-10*update/10000))-1
            coefficient = 2/(1+math.exp(-10*(update+1)/10000))-1
            transfer, domain_accuracy = cdan_entropy_loss(features, logits, self.discriminator,
                                                          source_count, entropy_coefficient, coefficient)
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
        weighted_transfer = weight*transfer
        # A zero-valued DA graph still creates zero .grad tensors on its
        # exclusive heads. Adam/SGD then apply weight decay (and momentum) to
        # those untrained parameters. Keep inactive heads outside the loss
        # graph so source-supervision warm-up actually preserves them.
        total = classification+weighted_transfer if weight > 0 else classification
        return total, {
            "classification": classification, "source_ce": ce_source,
            "auxiliary_source_ce": auxiliary_ce, "auxiliary_source_weight": getattr(self, "auxiliary_source_weight", 0.),
            "transfer": transfer, "weighted_transfer": weighted_transfer,
            "adaptation_graph_in_loss": weight > 0,
            "transfer_weight": weight, "grl_coefficient": coefficient,
            "entropy_grl_coefficient": entropy_coefficient if self.method == "cdan" else None,
            "domain_accuracy": domain_accuracy, "source_logits": ys, "target_logits": yt,
            "source_ce_unweighted": F.cross_entropy(ys, source["label"], label_smoothing=args.label_smoothing).detach() if log_moments else ce_source.detach(),
            "domain_feature_moments": {"before_bottleneck": domain_moments(fused_for_diagnostics, source_count),
                                       "after_bottleneck": domain_moments(features, source_count)} if log_moments else None,
            "method_feature_norms": {"source": float(features[:source_count].detach().norm(dim=1).mean()),
                                     "target": float(features[source_count:].detach().norm(dim=1).mean())} if log_moments else None,
            "logit_norms": {"source": float(ys.detach().norm(dim=1).mean()),
                            "target": float(yt.detach().norm(dim=1).mean())} if log_moments else None,
        }


def optimizer_groups(model, base_lr, ais_lr_multiplier=1.0, gate_lr_multiplier=1., gate_weight_decay=None):
    """Preserve the reference method's relative learning rates.

    Deep CORAL: backbone 1x / FC 10x. CDAN: backbone 1x / new layers and
    discriminator 10x. MDD/MCC: pretrained backbone .1x / new layers 1x.
    AIS and modality gates are new layers; none are hidden in pretrained groups.
    """
    all_pretrained = list(model.rgb_encoder.parameters())+list(model.nir_encoder.parameters())
    pretrained_ids = {id(p) for p in all_pretrained}
    pretrained = [p for p in all_pretrained if p.requires_grad]
    new = [p for p in model.parameters() if p.requires_grad and id(p) not in pretrained_ids]
    separate_gate = gate_lr_multiplier != 1. or gate_weight_decay is not None
    if gate_lr_multiplier <= 0 or (gate_weight_decay is not None and gate_weight_decay < 0):
        raise ValueError("Invalid gate learning-rate multiplier or weight decay")
    if separate_gate:
        gate = model.modality_logits
        new = [p for p in new if p is not gate]
    backbone_multiplier = 1.0 if model.method in {"deepcoral", "cdan"} else 0.1
    head_multiplier = 10.0 if model.method in {"deepcoral", "cdan"} else 1.0
    groups = [{"params": pretrained, "lr": base_lr*backbone_multiplier, "role": "pretrained_RGB_NIR"}]
    if ais_lr_multiplier != 1.0 and hasattr(model, "ais_encoder"):
        ais = list(model.ais_encoder.parameters())
        ais_ids = {id(p) for p in ais}
        groups.append({"params": [p for p in new if id(p) not in ais_ids],
                       "lr": base_lr*head_multiplier, "role": "new_layers" if separate_gate else "new_layers_gate"})
        groups.append({"params": ais, "lr": base_lr*head_multiplier*ais_lr_multiplier, "role": "AIS"})
    else:
        groups.append({"params": new, "lr": base_lr*head_multiplier,
                       "role": ("new_layers_AIS" if hasattr(model, "ais_encoder") else "new_layers") if separate_gate
                               else ("new_layers_AIS_gate" if hasattr(model, "ais_encoder") else "new_layers_gate")})
    if separate_gate and gate.requires_grad:
        group = {"params": [gate], "lr": base_lr*head_multiplier*gate_lr_multiplier, "role": "modality_gate"}
        if gate_weight_decay is not None:
            group["weight_decay"] = gate_weight_decay
        groups.append(group)
    return groups
