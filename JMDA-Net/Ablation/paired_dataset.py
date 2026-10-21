"""Strictly paired VIS/IR dataset with synchronized augmentation."""

import os
import hashlib
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from DataLoad import MultiModalDomainDataset


def filter_samples_by_original_labels(samples, allowed_original_labels=None):
    """Filter paired samples by original class id without changing label ids."""
    if allowed_original_labels is None:
        return list(samples)
    allowed = {int(label) for label in allowed_original_labels}
    if not allowed:
        raise ValueError("allowed_original_labels cannot be empty.")
    return [sample for sample in samples if int(sample["label"]) in allowed]


class PairedModalTransform:
    """Sample crop/flip once and apply it to both registered modalities."""

    def __init__(self, phase="train", val_augment=False):
        self.augment = phase == "train" or bool(val_augment)
        self.vis_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.ir_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def __call__(self, vis_img, ir_img):
        if self.augment:
            vis_img = TF.resize(vis_img, [256, 256])
            ir_img = TF.resize(ir_img, [256, 256])
            top, left, height, width = transforms.RandomCrop.get_params(
                vis_img,
                output_size=(224, 224),
            )
            vis_img = TF.crop(vis_img, top, left, height, width)
            ir_img = TF.crop(ir_img, top, left, height, width)
            if bool(torch.rand(()) < 0.5):
                vis_img = TF.hflip(vis_img)
                ir_img = TF.hflip(ir_img)
        else:
            vis_img = TF.resize(vis_img, [224, 224])
            ir_img = TF.resize(ir_img, [224, 224])

        return (
            self.vis_normalize(TF.to_tensor(vis_img)),
            self.ir_normalize(TF.to_tensor(ir_img)),
        )


class LabelFreePairedTargetDataset(torch.utils.data.Dataset):
    """Target-training VIS/IR pairs prepared without per-sample class labels."""

    def __init__(
        self,
        pair_list_path,
        num_classes,
        deterministic_transform=False,
    ):
        self.samples = []
        with open(pair_list_path, "r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) != 2 or not all(fields):
                    raise ValueError(
                        "Invalid label-free target pair at "
                        f"{pair_list_path}:{line_number}."
                    )
                vis_path, ir_path = fields
                if os.path.splitext(os.path.basename(vis_path))[0] != (
                    os.path.splitext(os.path.basename(ir_path))[0]
                ):
                    raise ValueError(
                        "Label-free target pair has mismatched filename stems at "
                        f"{pair_list_path}:{line_number}."
                    )
                if not os.path.isfile(vis_path) or not os.path.isfile(ir_path):
                    raise FileNotFoundError(
                        "Label-free target pair is missing an image at "
                        f"{pair_list_path}:{line_number}."
                    )
                self.samples.append({"vis": vis_path, "ir": ir_path})
        if not self.samples:
            raise ValueError(f"Label-free target pair list is empty: {pair_list_path}")

        self.labels = None
        self.unique_labels = []
        self.label_map = None
        self.num_classes = int(num_classes)
        self.include_label = False
        self.domain_label = 1
        self.phase = "train"
        self.val_augment = False
        self.deterministic_transform = bool(deterministic_transform)
        self.pairing_stats = {
            "source": "independent_label_free_pair_manifest",
            "selected_phase_pairs_before_filter": len(self.samples),
            "filtered_out_pairs": 0,
            "selected_phase_pairs": len(self.samples),
            "retained_original_labels": "withheld_from_training_dataset",
        }
        self.paired_transform = PairedModalTransform(
            phase="val" if self.deterministic_transform else "train",
            val_augment=False,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            vis_img = Image.open(sample["vis"]).convert("RGB")
            ir_img = Image.open(sample["ir"]).convert("L").convert("RGB")
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "Failed to load a label-free exact VIS/IR pair: "
                f"index={index}, vis={sample['vis']!r}, ir={sample['ir']!r}."
            ) from error
        vis_tensor, ir_tensor = self.paired_transform(vis_img, ir_img)
        return {
            "vis": vis_tensor,
            "ir": ir_tensor,
            "domain_label": self.domain_label,
        }


class PairedMultiModalDomainDataset(MultiModalDomainDataset):
    """Match modalities by filename stem and apply paired transformations."""

    def __init__(
        self,
        *args,
        allowed_original_labels=None,
        include_label=True,
        **kwargs,
    ):
        root_dir = args[0] if args else kwargs["root_dir"]
        domain_type = kwargs.get(
            "domain_type",
            args[1] if len(args) > 1 else "source",
        )
        phase = kwargs.get("phase", args[2] if len(args) > 2 else "train")
        global_label_map = kwargs.get("global_label_map")
        super().__init__(*args, **kwargs)
        self.samples, self.pairing_stats = self._build_consistent_pairs(
            root_dir=root_dir,
            domain_type=domain_type,
            phase=phase,
        )
        selected_before_filter = len(self.samples)
        self.samples = filter_samples_by_original_labels(
            self.samples,
            allowed_original_labels,
        )
        if not self.samples:
            raise ValueError(
                f"No allowed exact VIS/IR filename pairs found under "
                f"{root_dir!r}/{phase}."
            )
        self.labels = [sample["label"] for sample in self.samples]
        self.unique_labels = sorted(np.unique(self.labels))
        if global_label_map is None:
            self.label_map = {
                original_label: index
                for index, original_label in enumerate(self.unique_labels)
            }
        else:
            self.label_map = dict(global_label_map)
        unknown_labels = sorted(set(map(int, self.unique_labels)) - set(self.label_map))
        if unknown_labels:
            raise ValueError(
                "Dataset contains labels absent from global_label_map: "
                f"{unknown_labels}"
            )
        self.num_classes = len(self.label_map)
        self.include_label = bool(include_label)
        self.pairing_stats["selected_phase_pairs_before_filter"] = selected_before_filter
        self.pairing_stats["filtered_out_pairs"] = selected_before_filter - len(self.samples)
        self.pairing_stats["selected_phase_pairs"] = len(self.samples)
        self.pairing_stats["retained_original_labels"] = list(map(int, self.unique_labels))
        if not self.include_label:
            # A UDA optimizer only needs paired image paths. Remove per-sample
            # target truth from the live dataset object after initialization,
            # rather than merely omitting it from the collated batch. The
            # separately constructed oracle-support probe (when explicitly
            # requested) is the only training-domain object allowed to retain
            # target labels.
            self.samples = [
                {
                    key: value
                    for key, value in sample.items()
                    if key != "label"
                }
                for sample in self.samples
            ]
            self.labels = None
            self.unique_labels = []
        transform_phase = "val" if self.deterministic_transform else self.phase
        transform_val_augment = (
            False if self.deterministic_transform else self.val_augment
        )
        self.paired_transform = PairedModalTransform(
            phase=transform_phase,
            val_augment=transform_val_augment,
        )

    @staticmethod
    def _image_map(directory):
        valid_extensions = (".jpg", ".jpeg", ".png")
        return {
            os.path.splitext(filename)[0]: os.path.join(directory, filename)
            for filename in sorted(os.listdir(directory))
            if filename.lower().endswith(valid_extensions)
        }

    @staticmethod
    def _modal_dir(root_dir, domain_type, phase, class_name, modality):
        if domain_type == "source":
            return os.path.join(root_dir, phase, modality, class_name)
        return os.path.join(root_dir, phase, class_name, modality)

    def _build_consistent_pairs(self, root_dir, domain_type, phase):
        """Recover pairs from the train/val union, then split pairs together.

        The supplied dataset split assigned VIS and IR files independently.
        Consequently, positional pairing mixed observations and the two
        modalities of one vessel could land on opposite sides of train/val.
        We reconstruct exact filename-stem pairs from the union and use a
        stable hash to choose the original per-class validation count.
        """
        if phase not in {"train", "val"}:
            raise ValueError(f"Unsupported phase for paired split: {phase!r}")

        class_names = set()
        for split in ("train", "val"):
            if domain_type == "source":
                class_root = os.path.join(root_dir, split, "可见光")
            else:
                class_root = os.path.join(root_dir, split)
            if os.path.isdir(class_root):
                class_names.update(
                    name
                    for name in os.listdir(class_root)
                    if os.path.isdir(os.path.join(class_root, name))
                )

        samples = []
        stats = {
            "union_vis_files": 0,
            "union_ir_files": 0,
            "matched_pairs_all": 0,
            "selected_phase_pairs": 0,
            "dropped_unpaired_vis": 0,
            "dropped_unpaired_ir": 0,
        }
        for class_name in sorted(class_names):
            vis_by_stem = {}
            ir_by_stem = {}
            raw_val_vis = {}
            raw_val_ir = {}
            for split in ("train", "val"):
                vis_dir = self._modal_dir(
                    root_dir, domain_type, split, class_name, "可见光"
                )
                ir_dir = self._modal_dir(
                    root_dir, domain_type, split, class_name, "红外"
                )
                split_vis = self._image_map(vis_dir) if os.path.isdir(vis_dir) else {}
                split_ir = self._image_map(ir_dir) if os.path.isdir(ir_dir) else {}
                vis_by_stem.update(split_vis)
                ir_by_stem.update(split_ir)
                if split == "val":
                    raw_val_vis = split_vis
                    raw_val_ir = split_ir

            common_stems = set(vis_by_stem) & set(ir_by_stem)
            desired_val_count = min(
                len(raw_val_vis),
                len(raw_val_ir),
                len(common_stems),
            )
            ranked_stems = sorted(
                common_stems,
                key=lambda stem: hashlib.sha256(
                    f"{os.path.basename(root_dir)}:{class_name}:{stem}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            selected_stems = (
                ranked_stems[:desired_val_count]
                if phase == "val"
                else ranked_stems[desired_val_count:]
            )
            stats["union_vis_files"] += len(vis_by_stem)
            stats["union_ir_files"] += len(ir_by_stem)
            stats["matched_pairs_all"] += len(common_stems)
            stats["dropped_unpaired_vis"] += len(vis_by_stem) - len(common_stems)
            stats["dropped_unpaired_ir"] += len(ir_by_stem) - len(common_stems)
            stats["selected_phase_pairs"] += len(selected_stems)
            samples.extend(
                {
                    "vis": vis_by_stem[stem],
                    "ir": ir_by_stem[stem],
                    "label": int(class_name),
                }
                for stem in selected_stems
            )
        return samples, stats

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            vis_img = Image.open(sample["vis"]).convert("RGB")
            ir_img = Image.open(sample["ir"]).convert("L").convert("RGB")
        except (OSError, ValueError) as error:
            # Never replace a failed observation with the next dataset row.
            # That silently changes its class and invalidates the sample index,
            # pseudo label, VIS/IR pairing, and epoch-level SVD correspondence.
            raise RuntimeError(
                "Failed to load an exact VIS/IR pair: "
                f"index={idx}, vis={sample['vis']!r}, ir={sample['ir']!r}."
            ) from error

        vis_tensor, ir_tensor = self.paired_transform(vis_img, ir_img)
        result = {
            "vis": vis_tensor,
            "ir": ir_tensor,
            "domain_label": self.domain_label,
        }
        if self.include_label:
            try:
                label_id = self.label_map[sample["label"]]
            except KeyError as error:
                raise KeyError(
                    "Sample label is absent from label_map; refusing to silently "
                    f"remap original label {sample['label']} to class 0."
                ) from error
            result["label"] = torch.tensor(label_id, dtype=torch.long)
        return result


__all__ = [
    "PairedModalTransform",
    "PairedMultiModalDomainDataset",
    "LabelFreePairedTargetDataset",
    "filter_samples_by_original_labels",
]
