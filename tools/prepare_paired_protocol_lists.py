"""Create one exact VIS/IR-paired cohort shared by every comparison method."""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
TARGET_DIRS = {
    'backlight': '逆光',
    'rainy': '雨天',
    'foggy': '雾天',
    'dark': '黑天',
}


def image_map(directory):
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def modality_dir(domain_root, domain_type, split, class_name, modality):
    if domain_type == 'source':
        return domain_root / split / modality / class_name
    return domain_root / split / class_name / modality


def class_names(domain_root, domain_type):
    names = set()
    for split in ('train', 'val'):
        class_root = (
            domain_root / split / '可见光'
            if domain_type == 'source'
            else domain_root / split
        )
        if class_root.is_dir():
            names.update(
                child.name for child in class_root.iterdir() if child.is_dir()
            )
    return sorted(names, key=lambda name: int(name))


def exact_paired_split(domain_root, domain_type, phase):
    """Mirror PairedMultiModalDomainDataset's union/hash split exactly."""
    selected = []
    statistics = {
        'union_vis_files': 0,
        'union_ir_files': 0,
        'matched_pairs_all': 0,
        'selected_phase_pairs': 0,
        'dropped_unpaired_vis': 0,
        'dropped_unpaired_ir': 0,
    }
    for class_name in class_names(domain_root, domain_type):
        vis_by_stem, ir_by_stem = {}, {}
        raw_val_vis, raw_val_ir = {}, {}
        for split in ('train', 'val'):
            split_vis = image_map(
                modality_dir(domain_root, domain_type, split, class_name, '可见光')
            )
            split_ir = image_map(
                modality_dir(domain_root, domain_type, split, class_name, '红外')
            )
            vis_by_stem.update(split_vis)
            ir_by_stem.update(split_ir)
            if split == 'val':
                raw_val_vis, raw_val_ir = split_vis, split_ir

        common = set(vis_by_stem) & set(ir_by_stem)
        desired_val_count = min(len(raw_val_vis), len(raw_val_ir), len(common))
        ranked = sorted(
            common,
            key=lambda stem: hashlib.sha256(
                f'{domain_root.name}:{class_name}:{stem}'.encode('utf-8')
            ).hexdigest(),
        )
        stems = ranked[:desired_val_count] if phase == 'val' else ranked[desired_val_count:]
        statistics['union_vis_files'] += len(vis_by_stem)
        statistics['union_ir_files'] += len(ir_by_stem)
        statistics['matched_pairs_all'] += len(common)
        statistics['dropped_unpaired_vis'] += len(vis_by_stem) - len(common)
        statistics['dropped_unpaired_ir'] += len(ir_by_stem) - len(common)
        statistics['selected_phase_pairs'] += len(stems)
        selected.extend(
            (vis_by_stem[stem], ir_by_stem[stem], int(class_name))
            for stem in stems
        )
    return selected, statistics


def write_list(path, samples, label_map):
    lines = [
        f'{vis} {label_map[original_label]}'
        for vis, _, original_label in samples
    ]
    payload = ('\n'.join(lines) + '\n').encode('utf-8')
    path.write_bytes(payload)
    return {
        'rows': len(lines),
        'sha256': hashlib.sha256(payload).hexdigest(),
    }


def write_sda_subset(path, target_train, label_map, shots, seed):
    by_label = defaultdict(list)
    for sample in target_train:
        by_label[label_map[sample[2]]].append(sample)
    selected = []
    for label in sorted(by_label):
        candidates = sorted(by_label[label], key=lambda sample: str(sample[0]))
        rng = random.Random(seed * 1000 + label)
        count = min(shots, len(candidates))
        selected.extend(rng.sample(candidates, count))
    selected.sort(key=lambda sample: (label_map[sample[2]], str(sample[0])))
    return write_list(path, selected, label_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--targets', nargs='+', choices=tuple(TARGET_DIRS), required=True)
    parser.add_argument('--sda-shots', type=int, default=10)
    parser.add_argument('--sda-seed', type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = data_root / '晴天'

    source_train, source_train_stats = exact_paired_split(
        source_root, 'source', 'train'
    )
    source_val, source_val_stats = exact_paired_split(
        source_root, 'source', 'val'
    )
    original_labels = sorted({sample[2] for sample in source_train + source_val})
    if len(original_labels) != 14:
        raise ValueError(
            f'Expected 14 source classes, found {original_labels}.'
        )
    label_map = {label: index for index, label in enumerate(original_labels)}

    manifest = {
        'data_root': str(data_root),
        'label_map': label_map,
        'cohort_rule': 'exact VIS/IR stem match over train+val union; stable hash split',
        'lists': {},
        'pairing_stats': {
            'sunny_train': source_train_stats,
            'sunny_val': source_val_stats,
        },
    }
    manifest['lists']['sunny_train.txt'] = write_list(
        output_dir / 'sunny_train.txt', source_train, label_map
    )

    for target in args.targets:
        root = data_root / TARGET_DIRS[target]
        target_train, train_stats = exact_paired_split(root, 'target', 'train')
        target_val, val_stats = exact_paired_split(root, 'target', 'val')
        manifest['pairing_stats'][f'{target}_train'] = train_stats
        manifest['pairing_stats'][f'{target}_val'] = val_stats
        manifest['lists'][f'{target}_train.txt'] = write_list(
            output_dir / f'{target}_train.txt', target_train, label_map
        )
        manifest['lists'][f'{target}_val.txt'] = write_list(
            output_dir / f'{target}_val.txt', target_val, label_map
        )
        sda_name = f'{target}_sda_{args.sda_shots}shot_seed{args.sda_seed}.txt'
        manifest['lists'][sda_name] = write_sda_subset(
            output_dir / sda_name,
            target_train,
            label_map,
            args.sda_shots,
            args.sda_seed,
        )

    manifest_path = output_dir / 'paired_protocol_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'[OK] prepared exact-paired protocol lists: {output_dir}')
    for name, details in sorted(manifest['lists'].items()):
        print(f"  {name}: rows={details['rows']} sha256={details['sha256'][:12]}")


if __name__ == '__main__':
    main()
