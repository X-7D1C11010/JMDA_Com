"""
Generate the SHARED SDA (supervised domain adaptation) target label subset lists.

SDA protocol requirement: every SDA method must train on EXACTLY the same
source labels and the same target labeled subset, with the same number of
labels per class. This script draws that subset once, deterministically, from
data_lists/<target>_train.txt and writes:

    data_lists/<target>_sda_<N>shot_seed<S>.txt

so that every SDA training script (DANN-SDA, ..., JMDA-Net-SDA, Joint
Training, Target-only Oracle) loads the identical list file.

Selection rules:
  * fixed RNG seed (default 42), per-class sampling of N samples;
  * samples that have an IR partner (canonical JMDA-Net pairing) are
    preferred, so the same subset is usable by the multimodal methods
    (MODfinity-SDA / Boomda-SDA / JMDA-Net-SDA) without dropping entries;
  * if a class has fewer than N usable samples, all of them are taken and the
    shortfall is reported.

The file is deterministic: re-running with the same seed/N reproduces the
same subset byte-for-byte.

Usage:
    python tools/make_sda_lists.py --n-per-class 10 --seed 42
"""
import os
import argparse
import random

IMG_EXT = ('.jpg', '.jpeg', '.png')


def has_ir_partner(vis_path):
    """Canonical JMDA-Net pairing: within the same class folder, sorted(VIS)
    and sorted(IR) files are paired by index."""
    vis_dir = os.path.dirname(vis_path)
    ir_dir = vis_dir.replace('可见光', '红外')
    if not os.path.isdir(ir_dir):
        return False
    name = os.path.basename(vis_path)
    vis_files = sorted(f for f in os.listdir(vis_dir)
                       if f.lower().endswith(IMG_EXT))
    ir_files = sorted(f for f in os.listdir(ir_dir)
                      if f.lower().endswith(IMG_EXT))
    if name not in vis_files:
        return False
    return vis_files.index(name) < len(ir_files)


def read_list(list_path):
    per_class = {}
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            per_class.setdefault(int(parts[1]), []).append(parts[0])
    return per_class


def main():
    parser = argparse.ArgumentParser(description='Generate shared SDA target label subset lists')
    parser.add_argument('--data-dir', type=str, default=r'd:\Code\JMDA_Com\data_lists')
    parser.add_argument('--n-per-class', type=int, default=10,
                        help='Number of labeled target samples per class (SDA label budget)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--targets', nargs='+',
                        default=['backlight', 'rainy', 'foggy', 'dark'])
    args = parser.parse_args()

    summary = []
    for target in args.targets:
        src_list = os.path.join(args.data_dir, f'{target}_train.txt')
        out_list = os.path.join(
            args.data_dir, f'{target}_sda_{args.n_per_class}shot_seed{args.seed}.txt')
        per_class = read_list(src_list)

        chosen_lines = []
        report = []
        for c in sorted(per_class.keys()):
            pool = sorted(per_class[c])                       # deterministic order
            with_ir = [p for p in pool if has_ir_partner(p)]
            rng = random.Random(args.seed * 1000 + c)         # per-class deterministic RNG
            if len(with_ir) >= args.n_per_class:
                chosen = rng.sample(with_ir, args.n_per_class)
                used = 'ir-paired'
            elif len(pool) >= args.n_per_class:
                chosen = rng.sample(pool, args.n_per_class)
                used = 'fallback(any)'
            else:
                chosen = list(pool)
                used = f'short(all {len(pool)})'
            report.append((c, len(chosen), used))
            for p in chosen:
                chosen_lines.append(f'{p} {c}')

        # Write sorted by label then path for readability (order irrelevant to training)
        chosen_lines.sort(key=lambda l: (int(l.rsplit(' ', 1)[1]), l.rsplit(' ', 1)[0]))
        with open(out_list, 'w', encoding='utf-8') as f:
            f.write('\n'.join(chosen_lines) + '\n')

        total = len(chosen_lines)
        n_classes = len(report)
        print(f'[{target}] -> {os.path.basename(out_list)}')
        print(f'    classes={n_classes}, total labeled samples={total}, '
              f'budget={args.n_per_class}/class')
        for c, n, used in report:
            print(f'    class {c:2d}: {n:3d} samples ({used})')
        summary.append((target, out_list, total))

    print('\nSDA label subsets (identical for ALL SDA methods):')
    for target, path, total in summary:
        print(f'  {target:10s}: {path} ({total} samples)')


if __name__ == '__main__':
    main()
