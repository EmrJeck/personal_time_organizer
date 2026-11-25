"""Render thesis figures from the audited analysis, without rerunning methods."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ROOT/'experiments/.plot_packages'
if PACKAGES.exists():
    sys.path.insert(0, str(PACKAGES))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'experiments/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render():
    source = ROOT/'experiments/results_v1/analysis'
    result = json.loads((source/'analysis.json').read_text(encoding='utf-8'))
    manifest = json.loads((source/'analysis_manifest.json').read_text(encoding='utf-8'))
    if sha(source/'analysis.json') != manifest['output_checksums']['analysis.json']:
        raise ValueError('Analysis checksum mismatch')
    target = ROOT/'docs'
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12,
                         'axes.labelsize': 12, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.dpi': 300,
                         'axes.unicode_minus': True})
    formatter = FuncFormatter(lambda value, position: f'{value:g}'.replace('.', ','))
    summary = result['summary']
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    fig.subplots_adjust(left=.34, right=.97, bottom=.19, top=.97)
    values = np.array([row['q_mean'] for row in summary])
    lower = np.array([row['q_ci_low'] for row in summary])
    upper = np.array([row['q_ci_high'] for row in summary])
    labels = ['Найближчий строк', 'Лінійна оцінка', 'Лінійна оцінка\nз історією',
              'Нечітка оцінка', 'Запропонований\nметод']
    y = np.arange(len(summary))
    colors = ['#7C8793']*4 + ['#235C83']
    ax.barh(y, values, height=.62, color=colors, zorder=2)
    ax.errorbar(values, y, xerr=[values-lower, upper-values], fmt='none',
                ecolor='#111111', elinewidth=1.15, capsize=3.5, zorder=3)
    for row, mean, high in zip(y, values, upper):
        ax.text(high+.7, row, f'{mean:.2f}'.replace('.', ','), va='center', fontsize=12)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 44)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel('Модельований показник Q, %')
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(axis='x', color='#D8DDE2', linewidth=.6, zorder=0)
    ax.set_axisbelow(True)
    outputs = []
    for ext in ('png', 'svg'):
        path = target/f'figure_3_5_algorithm_comparison.{ext}'
        fig.savefig(path, facecolor='white')
        outputs.append(path)
    plt.close(fig)

    pairs = [row for row in result['paired_comparisons'] if row['scope'] == 'condition_history']
    order = [(condition, n) for condition in ('stable', 'no_time_effect', 'drift') for n in (0, 1, 20)]
    labels = ['Сталі звички, N = 0', 'N = 1', 'N = 20',
              'Без часового ефекту, N = 0', 'N = 1', 'N = 20',
              'Зміна звичок, N = 0', 'N = 1', 'N = 20']
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.75), sharey=True)
    fig.subplots_adjust(left=.34, right=.98, top=.90, bottom=.18, wspace=.15)
    for ax, right, title, xlabel in zip(axes, ('fuzzy', 'weighted_history'),
            ('а) Внесок історії', 'б) Внесок нечіткої оцінки'),
            ('Повний – нечіткий, в. п.', 'Повний – лінійний\nз історією, в. п.')):
        for start in (0, 6):
            ax.axhspan(start-.5, start+2.5, color='#F1F3F5', zorder=0)
        ax.axvline(0, color='#333333', linestyle='--', linewidth=.9, zorder=1)
        for i, (condition, n) in enumerate(order):
            row = next(r for r in pairs if r['condition'] == condition and r['history_n'] == n and r['right'] == right)
            mean, low, high = (row[k] for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp'))
            color = {'stable': '#235C83', 'no_time_effect': '#606870', 'drift': '#A54D26'}[condition]
            marker = {'stable': 'o', 'no_time_effect': 's', 'drift': '^'}[condition]
            ax.errorbar(mean, i, xerr=[[max(0, mean-low)], [max(0, high-mean)]],
                        fmt=marker, color=color, ecolor=color, markersize=5.5,
                        capsize=3, elinewidth=1.2, zorder=3)
        ax.set_title(title, pad=10, fontsize=11.5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_xlim(-4.5, 4.5)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(axis='x', color='#D8DDE2', linewidth=.5, zorder=0)
        ax.set_yticks(range(9), labels)
        ax.tick_params(axis='y', length=0, labelsize=11)
    axes[0].set_ylim(8.6, -.6)
    for ext in ('png', 'svg'):
        path = target/f'figure_3_6_component_effects.{ext}'
        fig.savefig(path, facecolor='white')
        outputs.append(path)
    plt.close(fig)
    metadata = {'source_analysis_sha256': sha(source/'analysis.json'),
                'plot_script_sha256': sha(__file__), 'matplotlib': matplotlib.__version__,
                'numpy': np.__version__, 'png_dpi': 300,
                'intervals': 'Marginal 95% paired stratified percentile bootstrap; generator uncertainty only',
                'files': {p.name: sha(p) for p in outputs}}
    (source/'figures_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figures': [str(p) for p in outputs]}, ensure_ascii=True))


if __name__ == '__main__':
    render()
