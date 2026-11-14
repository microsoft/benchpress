"""Plot main-experiment errors for BenchPress and its offsets-only ablation."""

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.io_utils import load_json, write_json_atomic
from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET, CHARCOAL, GRAY, PROBE_COST_AWARE_STYLE,
    PROBE_COST_UNAWARE_STYLE, apply_single, save_fig,
)
import matplotlib.pyplot as plt


STEM = 'bp_main_probe_offsets_medape'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=('medape', 'both'), default='medape')
    args = parser.parse_args()
    metrics = ('medae', 'medape') if args.metric == 'both' else ('medape',)
    stem = 'bp_main_probe_offsets_both_row' if args.metric == 'both' else STEM
    panels = (
        ('medae_any', '(a) Cost-unaware greedy', PROBE_COST_UNAWARE_STYLE),
        ('medae_low_cost', '(b) Cost-aware greedy', PROBE_COST_AWARE_STYLE),
    )
    apply_single()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    if args.metric == 'both':
        fig, axes = plt.subplots(1, 4, figsize=(23.8, 5.2))
        fig.subplots_adjust(left=0.045, right=0.992, bottom=0.19, top=0.70, wspace=0.30)
        fig.suptitle('Main experiment: probe selection and evaluation on 84 models',
                     fontsize=24, color=CHARCOAL, y=0.985)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), sharey=True)
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.75, wspace=0.13)
    plotted, sources = [], []
    for metric_index, metric in enumerate(metrics):
        for panel_index, (ordering, title, policy_style) in enumerate(panels):
            ax = axes[metric_index * 2 + panel_index]
            data_by_predictor = {}
            for predictor, suffix in (('benchpress', ''), ('offsets_only', '_rank0')):
                filename = f'fixed_order_{ordering}{suffix}_hidden_only.json.gz'
                path = os.path.join(HERE, 'results', filename)
                data = load_json(path)
                rows = sorted(data['summary_by_k'], key=lambda row: row['k'])
                if [row['k'] for row in rows] != list(range(1, 11)):
                    raise ValueError(f'Expected budgets 1 through 10: {filename}')
                if data['config']['n_models'] != 84:
                    raise ValueError(f'Expected the main 84-model experiment: {filename}')
                data_by_predictor[predictor] = data
                with open(path, 'rb') as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
                sources.append({'path': os.path.relpath(path, HERE), 'sha256': digest,
                                'manifest': data['manifest']})
            bp_rows = data_by_predictor['benchpress']['summary_by_k']
            offset_rows = data_by_predictor['offsets_only']['summary_by_k']
            for bp, offsets in zip(bp_rows, offset_rows):
                if bp['k'] != offsets['k'] or bp['probe_ids'] != offsets['probe_ids']:
                    raise ValueError(f'Probe ordering mismatch: {ordering}')
                if bp['hidden_only']['n'] != offsets['hidden_only']['n']:
                    raise ValueError(f'Coverage mismatch: {ordering}, k={bp["k"]}')
                plotted.append({'ordering': ordering, 'requested_probes': bp['k'],
                                'test_cells': bp['hidden_only']['n'],
                                **({'metric': metric} if len(metrics) == 2 else {}),
                                'benchpress': bp['hidden_only'][metric],
                                'offsets_only': offsets['hidden_only'][metric]})
            for name, label, color, linestyle, marker in (
                ('benchpress', 'BenchPress', policy_style['color'], '-', 'o'),
                ('offsets_only', 'Offsets only', ANSWER_VIOLET, '--', 's'),
            ):
                rows = data_by_predictor[name]['summary_by_k']
                ax.plot([r['k'] for r in rows], [r['hidden_only'][metric] for r in rows],
                        label=label, color=color, linestyle=linestyle, marker=marker,
                        linewidth=2.6, markersize=7.2, markeredgewidth=1.1,
                        markeredgecolor='white', zorder=4 if name == 'benchpress' else 3)
            panel_title = title if metric_index == 0 else title.replace('(a)', '(c)').replace('(b)', '(d)')
            ax.set_title(panel_title, fontsize=23, pad=43, color=CHARCOAL)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.025), ncol=2,
                      frameon=False, fontsize=17, handlelength=1.65,
                      handletextpad=0.4, columnspacing=1.2, borderaxespad=0)
            ax.set_xlabel(r'Requested probes ($k$)', fontsize=22, labelpad=9)
            ax.set_xticks([1, 2, 4, 6, 8, 10] if len(metrics) == 2 else range(1, 11))
            ax.set_xlim(0.7, 10.3)
            if len(metrics) == 1:
                ax.set_ylim(7, 11)
                ax.set_yticks(range(7, 12))
            elif metric == 'medae':
                ax.set_ylim(4, 10)
                ax.set_yticks(range(4, 11))
            else:
                ax.set_ylim(7, 18)
                ax.set_yticks(range(8, 19, 2))
            if len(metrics) == 2:
                ax.set_ylabel('MedAE (score points)' if metric == 'medae' else 'MedAPE (%)',
                              fontsize=21, labelpad=9)
            ax.tick_params(axis='both', labelsize=18)
            ax.grid(axis='y', color=GRAY, linewidth=0.75, alpha=0.22)
    if len(metrics) == 1:
        axes[0].set_ylabel('MedAPE (%)', fontsize=22, labelpad=12)
    save_fig(stem)
    figures = os.path.join(HERE, 'figures')
    write_json_atomic(os.path.join(figures, stem + '_data.json'), {
        'metric': ('pooled MedAPE on unrevealed observed cells with nonzero true score'
                   if len(metrics) == 1 else
                   'pooled MedAE and MedAPE on unrevealed cells; MedAPE excludes zero true scores'),
        'n_selection_models': 84, 'n_evaluated_models': 84,
        'sources': sources, 'data': plotted,
    }, indent=2)
    caption = (
        '**Main 84-model probe experiment, evaluated by MedAPE.** Probe orderings '
        'are selected using MedAE on the original April 30, 2026 matrix, exactly '
        'as for the main five-probe MedAE results. For each target model, the other '
        '83 models retain their observed scores, while only the target\'s available '
        'probe scores are revealed. BenchPress (rank 2) and offsets only (rank 0) '
        'are evaluated on the same unrevealed observed cells at each requested '
        'budget. MedAPE is the median of absolute prediction error divided by the '
        'absolute true score, multiplied by 100; lower is better, and zero true '
        'scores are excluded. Requested probes need not all have reported scores, '
        'and test cells change as more scores are revealed. This is the main '
        '84-model setup, separate from the 59/25 held-out-model validation.\n')
    if len(metrics) == 2:
        caption = caption.replace('evaluated by MedAPE', 'evaluated by MedAE and MedAPE')
        caption += 'Panels (a,b) show MedAE in score points; panels (c,d) show MedAPE (%). Both metrics use the same MedAE-selected probe orderings.\n'
    with open(os.path.join(figures, stem + '_caption.md'), 'w') as handle:
        handle.write(caption)


if __name__ == '__main__':
    main()
