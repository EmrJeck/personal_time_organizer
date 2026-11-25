"""Publication figures for the prespecified recency experiment."""
import hashlib
import json
import math
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render():
    source = ROOT/'experiments/results_v2/analysis'
    result = json.loads((source/'analysis.json').read_text(encoding='utf-8'))
    manifest = json.loads((source/'analysis_manifest.json').read_text(encoding='utf-8'))
    if sha(source/'analysis.json') != manifest['output_checksums']['analysis.json']:
        raise ValueError('Analysis checksum mismatch')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12,
                         'axes.labelsize': 12, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.dpi': 300})
    formatter = FuncFormatter(lambda value, position: f'{value:g}'.replace('.', ','))
    files = []
    def save(fig, name):
        for ext in ('png', 'svg'):
            path = ROOT/'docs'/f'{name}.{ext}'
            fig.savefig(path, facecolor='white')
            files.append(path)
        plt.close(fig)

    rows = result['summary']
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    fig.subplots_adjust(left=.39, right=.98, bottom=.16, top=.98)
    values = np.array([r['q_mean'] for r in rows])
    lo = np.array([r['q_ci_low'] for r in rows])
    hi = np.array([r['q_ci_high'] for r in rows])
    labels = ['Найближчий строк', 'Лінійна оцінка', 'Лінійна з історією', 'Нечітка оцінка',
              'Попередній повний\nметод', 'Лінійна з давністю\nісторії', 'Оновлений повний\nметод']
    colors = ['#7C8793']*6+['#235C83']
    y = np.arange(len(rows))
    ax.barh(y, values, height=.63, color=colors, zorder=2)
    ax.errorbar(values, y, xerr=[values-lo, hi-values], fmt='none', ecolor='#202020',
                elinewidth=1.1, capsize=3.5, zorder=3)
    for i, mean, high in zip(y, values, hi):
        ax.text(high+.65, i, f'{mean:.2f}'.replace('.', ','), va='center', fontsize=12)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(44, math.ceil((max(hi)+5)/5)*5))
    ax.set_xlabel('Модельований показник Q, %')
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(axis='x', color='#D8DDE2', linewidth=.6)
    ax.set_axisbelow(True)
    save(fig, 'figure_3_5_algorithm_comparison_v2')

    conditions = ['stable', 'no_time_effect', 'unseen_drift', 'observed_drift_7',
                  'observed_drift_14', 'observed_drift_28']
    labels = ['Сталі звички', 'Без часового ефекту', 'Зміна без нових\nспостережень',
              'Зміна 7 днів тому', 'Зміна 14 днів тому', 'Зміна 28 днів тому']
    pairs = [r for r in result['paired_comparisons'] if r['scope']=='condition_history'
             and r['right']=='personalized' and r['history_n'] in (1, 20)]
    extent = math.ceil(max(abs(r[k]) for r in pairs for k in ('ci_low_pp','ci_high_pp'))+.35)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.25), sharey=True)
    fig.subplots_adjust(left=.32, right=.98, bottom=.20, top=.90, wspace=.16)
    for ax, n in zip(axes, (1, 20)):
        ax.axvline(0, color='#333333', linewidth=.9, linestyle='--')
        for i, condition in enumerate(conditions):
            if i % 2 == 0:
                ax.axhspan(i-.5, i+.5, color='#F1F3F5', zorder=0)
            row = next(r for r in pairs if r['condition']==condition and r['history_n']==n)
            mean, low, high = [row[k] for k in ('difference_pp','ci_low_pp','ci_high_pp')]
            ax.errorbar(mean, i, xerr=[[max(0,mean-low)],[max(0,high-mean)]],
                        fmt='o', color='#235C83', markersize=5.5, capsize=3,
                        elinewidth=1.2, zorder=3)
        ax.set_title(f'N = {n}', pad=10)
        ax.set_xlabel('Новий – попередній\nметод, в. п.', fontsize=11)
        ax.set_xlim(-extent, extent)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(axis='x', color='#D8DDE2', linewidth=.5)
        ax.set_yticks(range(6), labels)
        ax.tick_params(axis='y', length=0, labelsize=11)
    axes[0].set_ylim(5.6, -.6)
    save(fig, 'figure_3_6_recency_effect_v2')
    metadata = {'source_analysis_sha256': sha(source/'analysis.json'),
                'plot_script_sha256': sha(__file__), 'matplotlib': matplotlib.__version__,
                'numpy': np.__version__, 'png_dpi': 300,
                'intervals': 'Marginal 95% paired stratified percentile bootstrap, generator uncertainty',
                'files': {p.name: sha(p) for p in files}}
    (source/'figures_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'files': [str(p) for p in files]}, ensure_ascii=True))


if __name__ == '__main__':
    render()
