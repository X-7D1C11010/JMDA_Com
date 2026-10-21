"""Extract per-epoch training curves from the comparison-method stdout logs.

纯后处理工具：只解析训练脚本已经写出的日志，产出逐 epoch 的 CSV 与收敛曲线图。
它不触碰任何方法的训练代码、超参或 checkpoint，属于"新增统计/日志"范畴。

用法::

    python tools/extract_epoch_curves.py \
        --logs-dir efficiency_results/uda_sunny/logs \
        --out-dir  efficiency_results/uda_sunny/epoch_curves

产出:
    <out-dir>/curves/<Method>_<target>_seed<seed>.csv   单次运行逐 epoch 明细
    <out-dir>/all_epochs.csv                            所有运行合并（长表，含 method/target/seed）
    <out-dir>/run_peak_final.csv                        每次运行的 峰值/最后一轮/差距
    <out-dir>/plots/<Method>_<target>.png               多种子均值±标准差的收敛曲线
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROGRESS_RE = re.compile(r'Epoch \[(\d+)/(\d+)\] iter \[\d+/\d+\]\s+(.*)$')
ACC_RE = re.compile(r'^Epoch \[(\d+)/\d+\] target val accuracy:\s*([0-9.]+)')
PRF_RE = re.compile(
    r'Val Precision \(Macro\):\s*([0-9.]+)\s*\|\s*'
    r'Val Recall \(Macro\):\s*([0-9.]+)\s*\|\s*'
    r'Val F1 \(Macro\):\s*([0-9.]+)\s*\|\s*'
    r'Val class coverage:\s*(\d+)/(\d+)'
)
KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9.]+)')
FINAL_RE = re.compile(r'^Final target val accuracy.*?:\s*([0-9.]+)')
PEAK_RE = re.compile(r'^Diagnostic peak target val accuracy.*?:\s*([0-9.]+)')
SPLIT_RE = re.compile(r'Source train:\s*(\d+),\s*target train:\s*(\d+),\s*target val:\s*(\d+)')
LOGNAME_RE = re.compile(r'^(?P<method>[A-Za-z0-9\-]+)_(?P<target>[a-z]+)_seed(?P<seed>\d+)\.log$')

META_COLUMNS = ('acc', 'precision_macro', 'recall_macro', 'f1_macro',
                'class_coverage', 'val_samples')


def detect_encoding(path):
    """PowerShell 5.1 用 `>` 重定向会写出 UTF-16LE(BOM) 日志，需要按 BOM 判定。"""
    with path.open('rb') as handle:
        head = handle.read(4)
    if head[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return 'utf-16'
    if head[:3] == b'\xef\xbb\xbf':
        return 'utf-8-sig'
    return 'utf-8'


def parse_log(path):
    """Return (rows, meta) for one training log.

    rows: list of dicts keyed by column name, one per epoch, ordered by epoch.
    meta: source/target/target-val sample counts plus the final/peak accuracy.
    """
    loss_sums = defaultdict(lambda: defaultdict(list))   # epoch -> key -> [values]
    accs = {}
    prf = {}
    meta = {}
    with path.open('r', encoding=detect_encoding(path), errors='replace') as handle:
        for line in handle:
            line = line.rstrip('\n')
            stripped = line.strip()
            progress = PROGRESS_RE.match(stripped)
            if progress:
                epoch = int(progress.group(1))
                for key, value in KV_RE.findall(progress.group(3)):
                    loss_sums[epoch][key].append(float(value))
                continue
            acc = ACC_RE.match(stripped)
            if acc:
                accs[int(acc.group(1))] = float(acc.group(2))
                continue
            prf_match = PRF_RE.search(stripped)
            if prf_match:
                epoch_match = re.match(r'^Epoch \[(\d+)/', stripped)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    prf[epoch] = (float(prf_match.group(1)), float(prf_match.group(2)),
                                  float(prf_match.group(3)), int(prf_match.group(4)),
                                  int(prf_match.group(5)))
                continue
            split = SPLIT_RE.search(stripped)
            if split:
                meta['source_train'] = int(split.group(1))
                meta['target_train'] = int(split.group(2))
                meta['target_val'] = int(split.group(3))
                continue
            final = FINAL_RE.match(stripped)
            if final:
                meta['reported_final_acc'] = float(final.group(1))
                continue
            peak = PEAK_RE.match(stripped)
            if peak:
                meta['reported_peak_acc'] = float(peak.group(1))
                continue

    loss_keys = sorted({key for per_epoch in loss_sums.values() for key in per_epoch})
    rows = []
    for epoch in sorted(accs):
        row = {'epoch': epoch, 'acc': accs[epoch]}
        if epoch in prf:
            row['precision_macro'], row['recall_macro'], row['f1_macro'] = prf[epoch][:3]
            row['class_coverage'], row['val_samples'] = prf[epoch][3:]
        for key in loss_keys:
            values = loss_sums.get(epoch, {}).get(key)
            if values:
                row[key] = statistics.fmean(values)
        rows.append(row)
    return rows, meta


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plots(series, plots_dir, metric):
    """scratch: 每个 (method,target) 画一张 准确率 + 训练损失 的双面板图。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f'[skip plots] matplotlib unavailable: {exc}')
        return 0

    plots_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for (method, target), runs in sorted(series.items()):
        max_epoch = max(len(run['acc']) for run in runs)
        figure, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))

        for metric_name, axis, title in ((
                'acc', ax_acc, 'target val accuracy'),):
            _plot_band(axis, runs, metric_name, max_epoch)
            axis.set_title(f'{method} -> {target}: {title}')
            axis.set_xlabel('epoch')
            axis.set_ylabel(title)
            axis.set_ylim(0.0, 1.0)
            axis.grid(alpha=0.3)

        for metric_name in metric:
            _plot_band(ax_loss, runs, metric_name, max_epoch, label=metric_name)
        ax_loss.set_title(f'{method} -> {target}: training losses')
        ax_loss.set_xlabel('epoch')
        ax_loss.set_ylabel('loss')
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(fontsize=7)

        figure.tight_layout()
        figure.savefig(plots_dir / f'{method}_{target}.png', dpi=110)
        plt.close(figure)
        written += 1
    return written


def _plot_band(axis, runs, metric_name, max_epoch, label=None):
    import numpy as np
    matrix = np.full((len(runs), max_epoch), np.nan)
    for index, run in enumerate(runs):
        values = run.get(metric_name, [])
        matrix[index, :len(values)] = values
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    epochs = np.arange(1, max_epoch + 1)
    axis.plot(epochs, mean, label=label or metric_name, linewidth=1.6)
    axis.fill_between(epochs, mean - std, mean + std, alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--logs-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    logs = sorted(p for p in logs_dir.glob('*.log') if LOGNAME_RE.match(p.name))
    if not logs:
        raise SystemExit(f'no per-run logs matched in {logs_dir}')

    series = defaultdict(list)
    all_rows = []
    run_rows = []
    loss_keys = set()
    parsed = 0

    for log_path in logs:
        match = LOGNAME_RE.match(log_path.name)
        method, target, seed = match.group('method'), match.group('target'), int(match.group('seed'))
        rows, meta = parse_log(log_path)
        if not rows:
            print(f'[skip] no epoch rows in {log_path.name}')
            continue
        parsed += 1
        for row in rows:
            loss_keys.update(row)
        # 单次运行明细
        fields = ['epoch'] + [c for c in META_COLUMNS if c in rows[0]] \
            + sorted(k for k in loss_keys if k not in META_COLUMNS and k != 'epoch')
        write_csv(out_dir / 'curves' / f'{method}_{target}_seed{seed}.csv', rows, fields)
        for row in rows:
            all_rows.append({'method': method, 'target': target, 'seed': seed, **row})
        series[(method, target)].append({
            'seed': seed,
            'acc': [row['acc'] for row in rows],
            **{k: [row.get(k) for row in rows] for k in loss_keys if k != 'epoch'},
        })
        accs = [row['acc'] for row in rows]
        peak_epoch = max(range(len(accs)), key=lambda i: accs[i]) + 1
        run_rows.append({
            'method': method, 'target': target, 'seed': seed,
            'epochs': len(rows),
            'first_epoch_acc': accs[0],
            'peak_epoch': peak_epoch,
            'peak_acc': max(accs),
            'final_acc': accs[-1],
            'peak_minus_final': max(accs) - accs[-1],
            'source_train': meta.get('source_train', ''),
            'target_val_samples': meta.get('target_val', ''),
            'reported_final_acc': meta.get('reported_final_acc', ''),
            'reported_peak_acc': meta.get('reported_peak_acc', ''),
        })

    loss_columns = sorted(k for k in loss_keys if k not in META_COLUMNS and k != 'epoch')
    all_fields = ['method', 'target', 'seed', 'epoch'] \
        + [c for c in META_COLUMNS if any(c in r for r in all_rows)] + loss_columns
    write_csv(out_dir / 'all_epochs.csv', all_rows, all_fields)
    if run_rows:
        write_csv(out_dir / 'run_peak_final.csv', run_rows, list(run_rows[0]))

    plot_count = 0
    if not args.no_plots:
        plot_count = make_plots(series, out_dir / 'plots', loss_columns)

    print(f'parsed {parsed} logs from {logs_dir}')
    print(f'  per-run curves : {out_dir / "curves"}')
    print(f'  merged epochs  : {out_dir / "all_epochs.csv"}')
    print(f'  peak vs final  : {out_dir / "run_peak_final.csv"}')
    if plot_count:
        print(f'  plots          : {out_dir / "plots"} ({plot_count} files)')


if __name__ == '__main__':
    main()
