"""Fail-fast validation for the unified JMDA comparison launcher."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_paths import exact_ir_partner, remap_data_path  # noqa: E402
from class_names import load_class_names  # noqa: E402


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


def validate_manifest(list_dir, data_root, required_modality='paired'):
    manifest_path = list_dir / 'paired_protocol_manifest.json'
    if not manifest_path.is_file():
        fail(f'Paired-cohort manifest is missing: {manifest_path}')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f'Invalid paired-cohort manifest {manifest_path}: {exc}')
    expected_rules = {
        'paired': 'exact VIS/IR stem match over train+val union; stable hash split',
        'vis': 'all VIS images over train+val union; stable hash split (no IR pairing)',
    }
    if manifest.get('modality') != required_modality:
        fail(
            f'Manifest modality {manifest.get("modality")!r} does not match '
            f'required modality {required_modality!r}'
        )
    expected_rule = expected_rules[required_modality]
    if manifest.get('cohort_rule') != expected_rule:
        fail(
            'Unsupported paired-cohort rule: '
            f'{manifest.get("cohort_rule")!r}'
        )
    manifest_root = os.path.normcase(
        os.path.realpath(str(manifest.get('data_root', '')))
    )
    actual_root = os.path.normcase(os.path.realpath(str(data_root)))
    if manifest_root != actual_root:
        fail(
            'Paired-cohort manifest was generated for another data root: '
            f'{manifest.get("data_root")!r} != {str(data_root)!r}'
        )
    entries = manifest.get('lists')
    pair_entries = manifest.get('unlabeled_pair_lists')
    if not isinstance(entries, dict) or not entries:
        fail(f'No list fingerprints in paired-cohort manifest: {manifest_path}')
    if required_modality == 'paired' and (
        not isinstance(pair_entries, dict) or not pair_entries
    ):
        fail(
            f'No label-free target pair fingerprints in manifest: {manifest_path}'
        )
    pair_entries = pair_entries if isinstance(pair_entries, dict) else {}
    for group in (entries, pair_entries):
        for name, expected in group.items():
            path = list_dir / name
            if not path.is_file():
                fail(f'Manifest list is missing: {path}')
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected.get('sha256'):
                fail(f'List fingerprint mismatch (possibly stale/edited): {path}')
            row_count = sum(1 for line in raw.splitlines() if line.strip())
            if row_count != expected.get('rows'):
                fail(
                    f'List row count differs from manifest for {path}: '
                    f'{row_count} != {expected.get("rows")}'
                )
    print(
        f'[OK] {required_modality} cohort manifest: {manifest_path} '
        f'({len(entries)} labelled lists, {len(pair_entries)} label-free pair lists)'
    )
    return manifest


def representative_rows(rows):
    if len(rows) <= 25:
        return rows
    step = max(1, len(rows) // 20)
    selected = rows[::step]
    if selected[-1] != rows[-1]:
        selected.append(rows[-1])
    return selected


def validate_list(path, data_root, check_all, require_ir=True):
    if not path.is_file():
        fail(f'Required list is missing: {path}')
    rows = read_list(path)
    selected = rows if check_all else representative_rows(rows)
    root = os.path.realpath(str(data_root))
    missing = []
    missing_ir = []
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
        if (
            os.name != 'nt'
            and mapped == stored
            and ('\\' in stored or ':' in stored)
        ):
            unchanged.append(stored)
        if not os.path.isfile(mapped):
            missing.append(mapped)
        elif require_ir and exact_ir_partner(mapped) is None:
            missing_ir.append(mapped)
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
    if require_ir and missing_ir:
        fail(
            f'Exact-stem IR mate was not found across train/val for {path}: '
            f'{missing_ir[0]}'
        )
    print(
        f'[OK] {path.name}: {len(rows)} rows, checked {len(selected)} '
        + ('VIS paths and exact IR mates' if require_ir else 'VIS paths')
    )
    return rows


def validate_unlabeled_pair_list(path, data_root, check_all):
    if not path.is_file():
        fail(f'Required label-free target pair list is missing: {path}')
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip('\r\n')
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) != 2 or not all(fields):
                fail(
                    f'Invalid label-free pair row {line_number} in {path}: '
                    f'{line!r}'
                )
            rows.append(tuple(fields))
    if not rows:
        fail(f'Label-free target pair list is empty: {path}')
    selected = rows if check_all else representative_rows(rows)
    root = os.path.realpath(str(data_root))
    for stored_vis, stored_ir in selected:
        vis = os.path.realpath(remap_data_path(stored_vis))
        ir = os.path.realpath(remap_data_path(stored_ir))
        try:
            inside = (
                os.path.commonpath([root, vis]) == root
                and os.path.commonpath([root, ir]) == root
            )
        except ValueError:
            inside = False
        if not inside:
            fail(
                f'Label-free pair escaped JMDA_DATA_ROOT in {path}: '
                f'{stored_vis!r}, {stored_ir!r}'
            )
        if not os.path.isfile(vis) or not os.path.isfile(ir):
            fail(f'Label-free pair contains a missing file: {vis!r}, {ir!r}')
        exact_ir = exact_ir_partner(vis)
        if exact_ir is None or os.path.realpath(exact_ir) != ir:
            fail(
                f'Label-free pair is not an exact VIS/IR mate: {vis!r}, {ir!r}'
            )
    print(
        f'[OK] {path.name}: {len(rows)} label-free rows, checked '
        f'{len(selected)} exact pairs'
    )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--protocol',
        choices=(
            'sda', 'jmda_uda', 'jmda_oracle_shared_uda',
            'jmda_label_guided_da', 'uda', 'vlm'
        ),
                        required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--jmda-net-dir', required=True)
    parser.add_argument('--list-dir', required=True)
    parser.add_argument('--targets', nargs='+', choices=TARGETS, required=True)
    parser.add_argument('--required-modality', choices=('paired', 'vis'), default='paired')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--check-all-paths', action='store_true')
    parser.add_argument(
        '--min-free-gpu-memory-mib',
        type=int,
        default=None,
        help=(
            'Optional CUDA free-memory gate in MiB. The check is disabled '
            'when this argument is omitted.'
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.min_free_gpu_memory_mib is not None
        and args.min_free_gpu_memory_mib < 0
    ):
        fail('--min-free-gpu-memory-mib must be a non-negative integer')
    data_root = Path(args.data_root).expanduser().resolve()
    list_dir = Path(args.list_dir).expanduser().resolve()
    if not data_root.is_dir():
        fail(f'Data root does not exist: {data_root}')
    if os.environ.get('JMDA_DATA_ROOT', '').strip() != str(data_root):
        fail(
            'JMDA_DATA_ROOT and --data-root disagree: '
            f'{os.environ.get("JMDA_DATA_ROOT")!r} != {str(data_root)!r}'
        )
    manifest = validate_manifest(list_dir, data_root, args.required_modality)

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
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        mib = 1024 * 1024
        free_mib = free_bytes // mib
        total_mib = total_bytes // mib
        print(
            f'[OK] CUDA device: {torch.cuda.get_device_name(0)}; '
            f'free/total memory={free_mib}/{total_mib} MiB'
        )
        if args.min_free_gpu_memory_mib is not None:
            required_bytes = args.min_free_gpu_memory_mib * mib
            if free_bytes < required_bytes:
                fail(
                    'Insufficient free CUDA memory on the selected device: '
                    f'{free_mib} MiB available, '
                    f'{args.min_free_gpu_memory_mib} MiB required. '
                    'Stop competing GPU jobs or select another device with '
                    'CUDA_VISIBLE_DEVICES.'
                )
            print(
                '[OK] CUDA free-memory gate passed: '
                f'{free_mib} MiB available >= '
                f'{args.min_free_gpu_memory_mib} MiB required'
            )
    elif args.min_free_gpu_memory_mib is not None:
        fail(
            '--min-free-gpu-memory-mib was requested, but CUDA is not available'
        )

    lists = [list_dir / 'sunny_train.txt']
    for target in args.targets:
        lists.append(list_dir / f'{target}_val.txt')
        if args.protocol == 'sda':
            lists.append(
                list_dir / f'{target}_sda_10shot_seed42.txt'
            )
        elif args.protocol in (
            'jmda_uda', 'jmda_oracle_shared_uda',
            'jmda_label_guided_da', 'uda', 'vlm'
        ):
            lists.append(list_dir / f'{target}_train.txt')
    validated = {}
    for path in lists:
        validated[path.name] = validate_list(
            path, data_root, args.check_all_paths,
            require_ir=args.required_modality == 'paired',
        )
    validated_pairs = {}
    if args.protocol in ('jmda_oracle_shared_uda', 'jmda_label_guided_da'):
        for target in args.targets:
            pair_path = list_dir / f'{target}_train_pairs.tsv'
            pair_rows = validate_unlabeled_pair_list(
                pair_path,
                data_root,
                args.check_all_paths,
            )
            labelled_rows = validated[f'{target}_train.txt']
            labelled_vis = [row[0] for row in labelled_rows]
            pair_vis = [row[0] for row in pair_rows]
            if pair_vis != labelled_vis:
                fail(
                    f'{pair_path.name} does not preserve every target-training '
                    'sample and order from the audited paired cohort'
                )
            validated_pairs[target] = pair_rows

    source_labels = {
        label for _, label in validated['sunny_train.txt']
    }
    expected_labels = set(range(14))
    if source_labels != expected_labels:
        fail(
            'sunny_train.txt must cover labels 0..13 exactly; got '
            f'{sorted(source_labels)}'
        )
    for target in args.targets:
        training_name = (
            f'{target}_sda_10shot_seed42.txt'
            if args.protocol == 'sda'
            else f'{target}_train.txt'
        )
        if training_name not in validated:
            continue
        target_labels = {label for _, label in validated[training_name]}
        val_labels = {
            label for _, label in validated[f'{target}_val.txt']
        }
        support_record = manifest.get('target_class_support', {}).get(target)
        if not isinstance(support_record, dict):
            fail(
                f'Manifest is missing target_class_support for {target}'
            )
        original_support = support_record.get('original_labels')
        if not isinstance(original_support, list) or not original_support:
            fail(
                f'Manifest has invalid original-label support for {target}: '
                f'{original_support!r}'
            )
        global_label_map = {
            int(original): int(mapped)
            for original, mapped in manifest.get('label_map', {}).items()
        }
        try:
            mapped_support = {
                global_label_map[int(label)] for label in original_support
            }
        except KeyError as exc:
            fail(
                f'{target} support contains an original class absent from the '
                f'source label map: {exc}'
            )
        if target_labels != mapped_support or val_labels != mapped_support:
            fail(
                f'{target} train/val lists and manifest support disagree: '
                f'manifest={sorted(mapped_support)}, '
                f'train={sorted(target_labels)}, val={sorted(val_labels)}'
            )
        if args.protocol in ('jmda_oracle_shared_uda', 'jmda_label_guided_da'):
            pair_rows = validated_pairs.get(target, [])
            if len(pair_rows) != len(validated[training_name]):
                fail(
                    f'{target} label-free pair list dropped target samples: '
                    f'{len(pair_rows)} != {len(validated[training_name])}'
                )
        expected_compact_map = {
            str(label): index
            for index, label in enumerate(sorted(map(int, original_support)))
        }
        if support_record.get('compact_label_map') != expected_compact_map:
            fail(
                f'{target} compact label map is invalid: '
                f'{support_record.get("compact_label_map")!r}'
            )
        unknown = sorted(target_labels - source_labels)
        if unknown:
            fail(
                f'{training_name} contains labels absent from the source: {unknown}'
            )
        absent = sorted(source_labels - target_labels)
        relation = 'partial-set' if absent else 'closed-set'
        print(
            f'[AUDIT] {target}: {relation}, target classes='
            f'{len(target_labels)}/14, source-only labels={absent}, '
            f'original support={sorted(map(int, original_support))}, '
            f'compact map={expected_compact_map}'
        )

    if args.protocol in ('uda', 'vlm'):
        class_names_path = os.environ.get('JMDA_CLASS_NAMES_FILE', '').strip()
        names = load_class_names(class_names_path, expected_count=14)
        print(
            f'[OK] real CLIP class-name mapping: {class_names_path} '
            f'({len(names)} classes)'
        )

    if args.protocol == 'jmda_oracle_shared_uda':
        print(
            '[ORACLE] target-training class support was isolated in the '
            'preprocessing manifest; training receives only the selected '
            'original class ids. This diagnostic is not strict UDA.'
        )
    elif args.protocol == 'jmda_label_guided_da':
        print(
            '[LABEL-GUIDED] target-training labels are available only to '
            'Tensor pairing/SVD and class-conditional OT; target '
            'classification CE is disabled.'
        )

    if args.protocol in (
        'sda', 'jmda_uda', 'jmda_oracle_shared_uda',
        'jmda_label_guided_da'
    ):
        jmda = Path(args.jmda_net_dir).expanduser().resolve()
        required = (
            jmda / 'main_sda.py' if args.protocol == 'sda'
            else jmda / 'Ablation' / 'module_ablation.py'
        )
        if not required.is_file():
            fail(f'JMDA-Net entry point is missing: {required}')
        print(f'[OK] JMDA-Net entry: {required}')
        protocol_explanation = {
            'sda': 'SDA; shared labeled target subset is used',
            'jmda_uda': (
                'strict UDA; target_label_ratio=0, target_cls_weight=0, '
                'final checkpoint reporting'
            ),
            'jmda_oracle_shared_uda': (
                'ORACLE_SHARED_UDA; target labels are excluded from losses, '
                'and an independently prepared class-support manifest filters '
                'the source label space'
            ),
            'jmda_label_guided_da': (
                'LABEL_GUIDED_DA_NO_TARGET_CE; true target classes supervise '
                'Tensor pairing/SVD and class-conditional OT, while target '
                'classification CE remains zero'
            ),
        }[args.protocol]
        print(f'[OK] JMDA protocol semantics: {protocol_explanation}')

    print('[OK] server preflight passed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[PREFLIGHT FAILED] {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
