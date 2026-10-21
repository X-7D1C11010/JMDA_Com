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

from data_paths import remap_data_path


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
    present = set()
    for split in ('train', 'val'):
        list_path = os.path.join(data_dir, f'{target}_{split}.txt')
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f'missing list file: {list_path}')
        present.update(label for _, label in read_image_list(list_path))
    if not present:
        raise ValueError(f'no samples found for target domain {target}')
    class_ids = sorted(present)
    label_map = {old: new for new, old in enumerate(class_ids)}
    return class_ids, label_map


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


def format_class_space(class_ids, num_classes):
    """日志用：目标域类别集描述。"""
    return (f'UDA label space (target-driven): {num_classes} classes, '
            f'original ids {class_ids}, remapped to 0..{num_classes - 1}')


def format_dropped(name, dataset):
    """日志用：某个列表被裁剪掉的图像数。"""
    return (f'  {name}: kept {len(dataset)}, dropped {dataset.skipped} '
            f'(label outside target label space)')


__all__ = [
    'read_image_list',
    'uda_label_space',
    'UdaImageListDataset',
    'format_class_space',
    'format_dropped',
]
