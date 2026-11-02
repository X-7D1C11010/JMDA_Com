# -*- coding: utf-8 -*-
"""无监督域适应（UDA）方法共用的数据加载：以目标域类别集为准裁剪数据。

背景：本数据集中源域（晴天）含全部 14 类，而各目标域天气场景只含其中一部分
类别。若沿用全 14 类标签空间，源域里有相当一部分图像属于目标域根本不存在的
类别，无监督对齐会被迫把这些"目标域无法解释"的类簇也压到一起。

因此本项目统一口径为 **以目标域类别为准**：

* 目标域类别集 ``T`` = 目标域 train 与 val 列表中出现的原始类别 id；
* 源域与目标域的 train/val 都只保留标签属于 ``T`` 的图像；
* 标签重映射到 ``0..|T|-1``，分类头输出维度即 ``|T|``。

用法（训练脚本）::

    from uda_data import uda_label_space, UdaImageListDataset

    class_ids, label_map = uda_label_space(args.data_dir, args.target)
    num_classes = len(class_ids)
    src_ds = UdaImageListDataset(os.path.join(args.data_dir, 'sunny_train.txt'),
                                 train_tf, label_map)
    tgt_ds = UdaImageListDataset(os.path.join(args.data_dir, f'{args.target}_train.txt'),
                                 train_tf, label_map)

注意：``num_classes`` 因此随目标域变化，调用 ``classification_metrics`` 时必须
显式传入该值，否则默认的 14 类范围校验会报错。
"""
import os

from PIL import Image
from torch.utils.data import Dataset

try:
    from .data_paths import exact_ir_partner, remap_data_path
except ImportError:  # Training scripts add tools/ directly to sys.path.
    from data_paths import exact_ir_partner, remap_data_path


def read_image_list(list_path):
    """读取 ``路径 标签`` 列表文件，返回 ``[(path, int_label), ...]``（原始标签空间）。"""
    samples = []
    with open(list_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            path, label = line.split()
            samples.append((path, int(label)))
    return samples


def uda_label_space(data_dir, target):
    """目标域类别集 ``T`` 与 ``原始标签 -> 新标签`` 映射。

    类别集取目标域 train 与 val 两个列表中出现过的类别（本数据集中两者一致），
    这样 val 的标签一定落在 ``0..|T|-1`` 内。

    返回 ``(class_ids, label_map)``：``class_ids`` 为升序原始类别 id，
    ``label_map`` 把原始 id 压到连续下标。
    """
    split_support = {}
    for split in ('train', 'val'):
        list_path = os.path.join(data_dir, f'{target}_{split}.txt')
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f'missing list file: {list_path}')
        split_support[split] = {
            label for _, label in read_image_list(list_path)
        }
    present = split_support['train']
    if not present:
        raise ValueError(f'no samples found for target domain {target}')
    if split_support['val'] != present:
        only_train = sorted(present - split_support['val'])
        only_val = sorted(split_support['val'] - present)
        raise ValueError(
            f'target train/val class support differs for {target}: '
            f'train-only={only_train}, val-only={only_val}'
        )
    class_ids = sorted(present)
    label_map = {old: new for new, old in enumerate(class_ids)}
    return class_ids, label_map


def assert_training_label_space(source_dataset, target_dataset, class_ids):
    """Fail fast unless source and target training sets cover the same classes.

    Both datasets must already have been filtered with the same ``label_map``.
    This check guards against a stale/corrupt list or an accidental independent
    remapping in a method-specific loader.
    """
    expected = set(range(len(class_ids)))
    source_labels = {label for _, label in source_dataset.samples}
    target_labels = {label for _, label in target_dataset.samples}
    if source_labels != expected or target_labels != expected:
        raise ValueError(
            'source/target training label support mismatch after remapping: '
            f'expected={sorted(expected)}, source={sorted(source_labels)}, '
            f'target={sorted(target_labels)}, original_target_ids={class_ids}'
        )
    if source_dataset.label_map != target_dataset.label_map:
        raise ValueError('source and target datasets do not share one label map')
    return True


class UdaImageListDataset(Dataset):
    """按目标域类别集裁剪并重映射标签的列表数据集。

    ``label_map`` 为 ``None`` 时保持原始标签（不裁剪）。
    ``skipped`` 记录因不属于目标域类别集而被丢弃的图像数，供日志核对。
    """

    def __init__(self, list_path, transform=None, label_map=None):
        self.list_path = list_path
        self.transform = transform
        self.label_map = label_map
        self.samples = []
        self.skipped = 0
        for path, label in read_image_list(list_path):
            if label_map is not None and label not in label_map:
                self.skipped += 1
                continue
            new_label = label_map[label] if label_map is not None else label
            self.samples.append((remap_data_path(path), new_label))
        if not self.samples:
            raise ValueError(
                f'{list_path} has no sample inside the target label space'
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class PairedUdaImageListDataset(Dataset):
    """Target-label-space dataset returning aligned VIS+IR model inputs."""

    def __init__(self, list_path, transform=None, label_map=None):
        self.list_path = list_path
        self.transform = transform
        self.label_map = label_map
        self.samples = []
        self.skipped = 0
        self.dropped_unpaired = 0
        for path, label in read_image_list(list_path):
            if label_map is not None and label not in label_map:
                self.skipped += 1
                continue
            vis_path = remap_data_path(path)
            ir_path = exact_ir_partner(vis_path)
            if ir_path is None:
                self.dropped_unpaired += 1
                continue
            new_label = label_map[label] if label_map is not None else label
            self.samples.append(((vis_path, ir_path), new_label))
        if not self.samples:
            raise ValueError(
                f'{list_path} has no paired sample inside the target label space'
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (vis_path, ir_path), label = self.samples[index]
        vis = Image.open(vis_path).convert('RGB')
        ir = Image.open(ir_path).convert('L')
        inputs = self.transform(vis, ir) if self.transform else (vis, ir)
        return inputs, label


def format_class_space(class_ids, num_classes):
    """日志用：目标域类别集描述。"""
    return (f'UDA label space (target-driven): {num_classes} classes, '
            f'original ids {class_ids}, remapped to 0..{num_classes - 1}')


def format_dropped(name, dataset):
    """日志用：某个列表被裁剪掉的图像数。"""
    unpaired = getattr(dataset, 'dropped_unpaired', 0)
    return (f'  {name}: kept {len(dataset)}, dropped {dataset.skipped} '
            f'(label outside target label space), dropped_unpaired {unpaired}')


__all__ = [
    'read_image_list',
    'uda_label_space',
    'assert_training_label_space',
    'UdaImageListDataset',
    'PairedUdaImageListDataset',
    'format_class_space',
    'format_dropped',
]
