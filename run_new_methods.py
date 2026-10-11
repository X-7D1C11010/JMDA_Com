"""
Run the 4 newly added DA methods (MDD, MCC, Boomda, MODfinity) on all 4 target
domains with the same smoke protocol as the first batch (source=sunny, 3
epochs), parse the "Best target val accuracy" lines, and append a results
section to results_summary.txt.
"""
import re
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = r'D:\Anaconda\envs\pytorch\python.exe'
TARGETS = ['backlight', 'rainy', 'foggy', 'dark']
METHODS = [
    ('MDD', os.path.join(ROOT, 'MDD', 'train_mdd.py'), []),
    ('MCC', os.path.join(ROOT, 'MCC', 'train_mcc.py'), []),
    ('Boomda', os.path.join(ROOT, 'Boomda', 'train_boomda_jmda.py'), []),
    ('MODfinity', os.path.join(ROOT, 'Modfinity', 'train_modfinity.py'), []),
]
PATTERN = re.compile(r'Best target val accuracy \((\w+)\): ([0-9.]+)')


def run_one(name, script, target):
    cmd = [PYTHON, '-B', script, '--target', target, '--epochs', '3']
    print('=' * 60)
    print(f'Running {name} on target={target} (epochs=3)')
    print('=' * 60, flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    # stream the tail of the output for the log
    sys.stdout.write(proc.stdout[-2000:] if proc.stdout else '')
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr[-3000:] if proc.stderr else '')
        print(f'  FAILED (exit {proc.returncode})')
        return None
    matches = PATTERN.findall(proc.stdout or '')
    if not matches:
        print('  No result line found!')
        return None
    tgt, acc = matches[-1]
    print(f'  Result: {acc}')
    return float(acc)


def main():
    results = {}
    for name, script, extra in METHODS:
        results[name] = {}
        for target in TARGETS:
            acc = run_one(name, script, target)
            results[name][target] = acc

    # ---- append to results_summary.txt ----
    lines = ['', '', 'Newly added methods (second batch, same smoke protocol:',
             'source=sunny, 3 epochs, target-domain val accuracy)',
             'MODfinity is our reimplementation (no public code).',
             '=' * 80,
             'Method             backlight        rainy        foggy         dark       Average',
             '-' * 80]
    for name, _, _ in METHODS:
        row = results[name]
        vals = [row.get(t) for t in TARGETS]
        cells = '  '.join('----' if v is None else f'{v:.4f}' for v in vals)
        avg = sum(v for v in vals if v is not None) / max(1, sum(v is not None for v in vals))
        label = name if name != 'MODfinity' else 'MODfinity*'
        lines.append(f'{label:<20}{cells}  {avg:.4f}')
    lines.append('=' * 80)
    lines.append('Random baseline (14 classes): 0.0714   (* our reimplementation)')

    with open(os.path.join(ROOT, 'results_summary.txt'), 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\nAppended results to results_summary.txt')
    for name, _, _ in METHODS:
        print(name, results[name])


if __name__ == '__main__':
    main()
