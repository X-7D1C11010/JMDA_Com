"""Plot the five real foggy runs, without changing any training records."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=root / 'runs/weather_all/local_backlight_foggy_20261117_110225_632')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
        if name in fonts:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    rows = [r for r in csv.DictReader((args.run_dir / 'runs.csv').open(encoding='utf-8-sig'))
            if r['target'] == 'foggy' and r['method'] in ('dann', 'deepcoral')]
    history = {}
    for row in rows:
        directory = Path(row['result_file']).parent
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert result['protocol'] == 'LABEL_GUIDED_CLASS_ALIGNMENT'
        events = [json.loads(line) for line in (directory / 'evaluation.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        events = sorted((e for e in events if e['partition'] == 'target_val'), key=lambda e: e['epoch'])
        assert len(events) == result['completed_epochs'] == int(row['completed_epochs'])
        assert len({e['epoch'] for e in events}) == len(events)
        history[row['method'], int(row['seed'])] = (row, events)
    seeds = list(range(42, 47))
    assert set(history) == {(method, seed) for method in ('dann', 'deepcoral') for seed in seeds}
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    styles = ['-', '--', '-.', ':', '-']
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharex=True, sharey=True)
    for ax, method, title in zip(axes, ('dann', 'deepcoral'), ('DANN', 'Deep CORAL')):
        ax.axhspan(70, 77, color='#B8B8B8', alpha=.25, zorder=0)
        for seed, color, style in zip(seeds, colors, styles):
            row, events = history[method, seed]
            ax.plot([e['epoch'] for e in events], [100 * e['metrics']['accuracy'] for e in events],
                    color=color, linestyle=style, linewidth=1.7, alpha=.9,
                    label=f'seed {seed}（{len(events)}轮）')
        ax.set(title=title, xlabel='Epoch', xlim=(1, 40), ylim=(0, 102))
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
        ax.set_yticks(range(0, 101, 10))
        ax.grid(alpha=.2)
        ax.legend(loc='lower right', framealpha=.95, fontsize=9)
    axes[0].set_ylabel('目标验证集 ACC（%）')
    fig.suptitle('雾天：五次训练的逐轮验证准确率', fontsize=16)
    fig.text(.5, .015, '标签引导分支 · seed 42–46 · 灰色区域为70%–77%参考区间 · 曲线结束处为实际停止轮数', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(args.output_dir / 'foggy_accuracy_curves.png', dpi=180)
    fig.savefig(args.output_dir / 'foggy_accuracy_curves.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    offsets = (-.18, .18)
    for method, title, color, offset in zip(('dann', 'deepcoral'), ('DANN', 'Deep CORAL'), colors[:2], offsets):
        values = [float(history[method, seed][0]['ACC']) for seed in seeds]
        bars = ax.bar([i + offset for i in range(5)], values, width=.34, color=color, label=title)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=10)
    ax.set_xticks(range(5), [f'seed {seed}' for seed in seeds])
    ax.set(ylabel='目标验证集 ACC（%）', ylim=(0, 106), title='雾天：各seed原验证CE选轮结果')
    ax.set_yticks(range(0, 101, 10)); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.legend(loc='lower right')
    fig.text(.5, .015, '标签引导分支 · 使用原验证CE选轮成绩，非区间筛选或第5高值统计', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(args.output_dir / 'foggy_accuracy_by_seed.png', dpi=180)
    fig.savefig(args.output_dir / 'foggy_accuracy_by_seed.pdf')
    plt.close(fig)
    output = [{'method': method, 'seed': seed, 'epoch': event['epoch'], 'validation_ACC_percent': 100 * event['metrics']['accuracy'],
               'protocol': 'LABEL_GUIDED_CLASS_ALIGNMENT'}
              for (method, seed), (_, events) in history.items() for event in events]
    with (args.output_dir / 'plotted_accuracy.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    print(json.dumps({'runs': len(history), 'epoch_records': len(output), 'output': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
