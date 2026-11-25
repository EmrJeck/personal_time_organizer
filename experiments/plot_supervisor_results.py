"""Verified 300-dpi figures for the eight-method and 15-task comparisons."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'experiments/.plot_packages'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'experiments/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ORDER = ('edf', 'importance', 'weighted', 'weighted_history', 'fuzzy',
         'personalized', 'weighted_recent', 'personalized_recent')
LABELS = {'edf': 'Найближчий строк', 'importance': 'Лише важливість',
          'weighted': 'Лінійна оцінка', 'weighted_history': 'Лінійна з історією',
          'fuzzy': 'Нечітка оцінка', 'personalized': 'Повний метод\nбез давності',
          'weighted_recent': 'Лінійна з давністю',
          'personalized_recent': 'Повний метод\nіз давністю'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    folder = ROOT/'experiments'/name/'analysis'
    manifest = json.loads((folder/'analysis_manifest.json').read_text(encoding='utf-8'))
    if sha(folder/'analysis.json') != manifest['output_checksums']['analysis.json']:
        raise ValueError('Analysis checksum mismatch')
    return folder, json.loads((folder/'analysis.json').read_text(encoding='utf-8'))


def render():
    extended_folder, extended = load('results_v2_extended')
    supervisor_folder, supervisor = load('results_supervisor_15')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12,
                         'axes.labelsize': 12, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.dpi': 300})
    formatter = FuncFormatter(lambda value, position: f'{value:g}'.replace('.', ','))
    outputs = {}

    def save(fig, name, folder):
        files = []
        for ext in ('png', 'svg'):
            path = ROOT/'docs'/f'{name}.{ext}'
            fig.savefig(path, facecolor='white')
            files.append(path)
        plt.close(fig)
        outputs.setdefault(folder, []).extend(files)

    def color(method):
        return '#235C83' if method == 'personalized_recent' else '#A27943' if method == 'importance' else '#7C8793'

    lookup = {row['method']: row for row in extended['summary']}
    rows = [lookup[method] for method in ORDER]
    fig, ax = plt.subplots(figsize=(7.1, 4.9))
    fig.subplots_adjust(left=.36, right=.98, bottom=.14, top=.98)
    y = np.arange(len(rows))
    values = np.array([row['q_mean'] for row in rows])
    low = np.array([row['q_ci_low'] for row in rows])
    high = np.array([row['q_ci_high'] for row in rows])
    ax.barh(y, values, color=[color(method) for method in ORDER], height=.63, zorder=2)
    ax.errorbar(values, y, xerr=[values-low, high-values], fmt='none', ecolor='#202020',
                elinewidth=1.1, capsize=3.5, zorder=3)
    for position, mean, end in zip(y, values, high):
        ax.text(end+.65, position, f'{mean:.2f}'.replace('.', ','), va='center')
    ax.set_yticks(y, [LABELS[method] for method in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    ax.set_xlabel('Модельований показник Q, %')
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(axis='x', color='#D8DDE2', linewidth=.6)
    ax.set_axisbelow(True)
    save(fig, 'figure_3_5_algorithm_comparison_v3', extended_folder)

    main = ('edf', 'importance', 'personalized', 'personalized_recent')
    lookup = {row['method']: row for row in supervisor['summary'] if row['history_n'] == 20}
    rows = [lookup[method] for method in main]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), sharey=True)
    fig.subplots_adjust(left=.29, right=.98, bottom=.20, top=.88, wspace=.19)
    y = np.arange(len(main))
    for ax, mean_key, low_key, high_key, title, xlabel, limit, decimals in (
        (axes[0], 'p_mean', 'p_ci_low', 'p_ci_high', 'а) Розміщення задач', 'P, %', 110, 2),
        (axes[1], 'c_ratio_of_means', 'c_ci_low', 'c_ci_high', 'б) Цінність плану', 'C відносно EDF', 1.35, 3)):
        values = np.array([row[mean_key] for row in rows])
        low = np.array([row[low_key] for row in rows])
        high = np.array([row[high_key] for row in rows])
        ax.barh(y, values, color=[color(method) for method in main], height=.57, zorder=2)
        ax.errorbar(values, y, xerr=[values-low, high-values], fmt='none', ecolor='#202020',
                    elinewidth=1.1, capsize=3, zorder=3)
        for position, mean, end in zip(y, values, high):
            label_x = max(end+limit*.025, 1.035) if mean_key == 'c_ratio_of_means' else end+limit*.025
            ax.text(label_x, position, f'{mean:.{decimals}f}'.replace('.', ','),
                    va='center', fontsize=11)
        ax.set_xlim(0, limit)
        ax.set_title(title, pad=9)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(axis='x', color='#D8DDE2', linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_yticks(y, [LABELS[method] for method in main])
    axes[0].invert_yaxis()
    axes[1].axvline(1, linestyle='--', color='#303030', linewidth=.8, zorder=1)
    save(fig, 'figure_3_7_supervisor_15', supervisor_folder)

    for folder, files in outputs.items():
        metadata = {'source_analysis_sha256': sha(folder/'analysis.json'),
                    'plot_script_sha256': sha(__file__), 'matplotlib': matplotlib.__version__,
                    'numpy': np.__version__, 'png_dpi': 300,
                    'files': {str(path.relative_to(ROOT)): sha(path) for path in files},
                    'intervals': '95% percentile bootstrap; generator uncertainty only',
                    'supervisor_figure_methods': list(main), 'supervisor_history_n': 20}
        (folder/'figures_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'files': [str(p) for files in outputs.values() for p in files]}, ensure_ascii=True))


if __name__ == '__main__':
    render()
