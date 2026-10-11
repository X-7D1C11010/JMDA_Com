"""Fail-fast validation for the unified JMDA comparison launcher."""

import argparse
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from data_paths import remap_data_path  # noqa: E402


TARGETS = ('backlight', 'rainy', 'foggy', 'dark')
TARGET_DIRS = {
    'backlight': '逆光',
    'rainy': '雨天',
    'foggy': '雾天',
    'dark': '黑天',
}


def fail(message):
    raise RuntimeError(message)


def read_list(path):
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                stored_path, label = line.rsplit(maxsplit=1)
                int(label)
            except (TypeError, ValueError) as exc:
                fail(f'Invalid row {line_number} in {path}: {line!r} ({exc})')
            rows.append((stored_path, int(label)))
    if not rows:
        fail(f'List is empty: {path}')
    return rows


def representative_rows(rows):
    if len(rows) <= 25:
        return rows
    step = max(1, len(rows) // 20)
    selected = rows[::step]
    if selected[-1] != rows[-1]:
        selected.append(rows[-1])
    return selected


def validate_list(path, data_root, check_all):
    if not path.is_file():
        fail(f'Required list is missing: {path}')
    rows = read_list(path)
    selected = rows if check_all else representative_rows(rows)
    root = os.path.realpath(str(data_root))
    missing = []
    outside = []
    unchanged = []
    for stored, _ in selected:
        mapped = remap_data_path(stored)
        mapped_real = os.path.realpath(mapped)
        try:
            inside = os.path.commonpath([root, mapped_real]) == root
        except ValueError:
            inside = False
        if not inside:
            outside.append((stored, mapped))
        if mapped == stored and ('\\' in stored or ':' in stored):
            unchanged.append(stored)
        if not os.path.isfile(mapped):
            missing.append(mapped)
    if unchanged:
        fail(
            f'Windows paths were not remapped in {path}: '
            f'{unchanged[0]!r}. JMDA_DATA_ROOT={data_root}'
        )
    if outside:
        fail(
            f'Remapped path escaped JMDA_DATA_ROOT in {path}: '
            f'{outside[0][0]!r} -> {outside[0][1]!r}'
        )
    if missing:
        fail(f'Mapped image does not exist for {path}: {missing[0]}')
    print(f'[OK] {path.name}: {len(rows)} rows, checked {len(selected)} paths')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=('sda', 'jmda_uda', 'uda', 'vlm'),
                        required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--jmda-net-dir', required=True)
    parser.add_argument('--targets', nargs='+', choices=TARGETS, required=True)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--check-all-paths', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        fail(f'Data root does not exist: {data_root}')
    if os.environ.get('JMDA_DATA_ROOT', '').strip() != str(data_root):
        fail(
            'JMDA_DATA_ROOT and --data-root disagree: '
            f'{os.environ.get("JMDA_DATA_ROOT")!r} != {str(data_root)!r}'
        )

    sunny = data_root / '晴天'
    if not sunny.is_dir():
        fail(f'Source domain directory is missing: {sunny}')
    for target in args.targets:
        target_dir = data_root / TARGET_DIRS[target]
        if not target_dir.is_dir():
            fail(f'Target domain directory is missing: {target_dir}')

    import torch
    import torchvision
    print(f'[OK] torch={torch.__version__}, torchvision={torchvision.__version__}')
    if args.require_cuda and not torch.cuda.is_available():
        fail('CUDA is required but torch.cuda.is_available() is false')
    if torch.cuda.is_available():
        print(f'[OK] CUDA device: {torch.cuda.get_device_name(0)}')

    lists = [PROJECT / 'data_lists' / 'sunny_train.txt']
    for target in args.targets:
        lists.append(PROJECT / 'data_lists' / f'{target}_val.txt')
        if args.protocol == 'sda':
            lists.append(
                PROJECT / 'data_lists' / f'{target}_sda_10shot_seed42.txt'
            )
        elif args.protocol in ('uda', 'vlm'):
            lists.append(PROJECT / 'data_lists' / f'{target}_train.txt')
    for path in lists:
        validate_list(path, data_root, args.check_all_paths)

    if args.protocol in ('sda', 'jmda_uda'):
        jmda = Path(args.jmda_net_dir).expanduser().resolve()
        required = (
            jmda / 'main_sda.py' if args.protocol == 'sda'
            else jmda / 'Ablation' / 'module_ablation.py'
        )
        if not required.is_file():
            fail(f'JMDA-Net entry point is missing: {required}')
        print(f'[OK] JMDA-Net entry: {required}')

    print('[OK] server preflight passed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[PREFLIGHT FAILED] {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
