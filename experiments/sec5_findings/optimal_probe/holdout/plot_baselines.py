"""Plot the original 59/25 held-out comparison from saved baseline results."""

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


STEM = 'bp_heldout_probe_baselines'
CAPTION = (
    '**Predictor comparison on 25 held-out models.** Greedy probes are selected '
    'using 59 of the 84 models in the April 30, 2026 score matrix (split seed 42). '
    'For each held-out target, predictors use those 59 model rows plus the '
    'target\'s available scores among the first $k$ selected probes. The other '
    'held-out rows remain hidden. Curves show median absolute error (MedAE, '
    'lower is better) pooled over the target models\' remaining observed scores. '
    'BenchPress, offsets only (rank 0), and logit-space model mean use the same '
    'available probe scores and test cells at each budget. Requested probes '
    'need not all have reported scores; revealed probe cells are excluded from '
    'the error, so test cells change across budgets. All 25 target models are '
    'retained at every budget; the model-mean baseline falls back to the '
    'benchmark mean when a target has no available probe scores.\n'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=os.path.join(
        HERE, 'results', 'heldout_baselines_s42.json.gz'))
    parser.add_argument('--metric', choices=('medae', 'medape', 'both'), default='medae')
    parser.add_argument('--layout', choices=('grid', 'row'), default='grid')
    args = parser.parse_args()
    row_layout = args.layout == 'row' and args.metric == 'both'
    metrics = ('medae', 'medape') if args.metric == 'both' else (args.metric,)
    stem = STEM + {'medae': '', 'medape': '_medape', 'both': '_both'}[args.metric]
    if row_layout:
        stem += '_row'
    data = load_json(args.input)
    if data['config']['protocol'] != 'heldout_baselines_v2':
        raise ValueError('Expected the original held-out baseline protocol')
    panels = (
        ('cost_unaware_greedy', '(a) Cost-unaware greedy', PROBE_COST_UNAWARE_STYLE),
        ('cost_aware_greedy', '(b) Cost-aware greedy', PROBE_COST_AWARE_STYLE),
    )
    apply_single()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    if row_layout:
        fig, axes = plt.subplots(1, 4, figsize=(23.8, 5.2), squeeze=False)
        fig.subplots_adjust(left=0.045, right=0.992, bottom=0.19, top=0.70,
                            wspace=0.30)
        fig.suptitle('Held-out validation: 59 selection models / 25 test models',
                     fontsize=24, color=CHARCOAL, y=0.985)
    else:
        fig, axes = plt.subplots(
            len(metrics), 2, figsize=(12.6, 8.6 if len(metrics) == 2 else 5.1),
            sharex='col', sharey='row', squeeze=False)
        if len(metrics) == 2:
            fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.86,
                                wspace=0.13, hspace=0.36)
        else:
            fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.75,
                                wspace=0.13)
    plotted = []
    for row_index, metric in enumerate(metrics):
        for col_index, (ordering, title, policy_style) in enumerate(panels):
            ax = axes[0, row_index * 2 + col_index] if row_layout else axes[row_index, col_index]
            rows = sorted((s for s in data['summary'] if s['method'] == ordering),
                          key=lambda s: s['k'])
            if [s['k'] for s in rows] != list(range(1, 11)):
                raise ValueError(f'Expected every requested budget from 1 to 10: {ordering}')
            for row in rows:
                n = row['n_cells_in_prediction_payload']
                if any(row[m]['n'] != n for m in
                       ('benchpress', 'logit_offsets_only', 'logit_model_mean')):
                    raise ValueError(f'Predictor coverage mismatch: {ordering}, k={row["k"]}')
            series = (
                ('benchpress', 'BenchPress', policy_style['color'], '-', 'o'),
                ('logit_offsets_only', 'Offsets only', ANSWER_VIOLET, '--', 's'),
                ('logit_model_mean', 'Model mean', GRAY, ':', '^'),
            )
            for name, label, color, linestyle, marker in series:
                ax.plot([r['k'] for r in rows], [r[name][metric] for r in rows],
                        label=label, color=color, linestyle=linestyle, marker=marker,
                        linewidth=2.6, markersize=7.2, markeredgewidth=1.1,
                        markeredgecolor='white', zorder=4 if name == 'benchpress' else 3)
            panel_title = title if row_index == 0 else title.replace('(a)', '(c)').replace('(b)', '(d)')
            ax.set_title(panel_title, fontsize=23, pad=43 if row_index == 0 or row_layout else 12, color=CHARCOAL)
            if row_index == 0 or row_layout:
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.025), ncol=3,
                          frameon=False, fontsize=14.5 if row_layout else 15.5, handlelength=1.55,
                          handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
            if row_layout or row_index == len(metrics) - 1:
                ax.set_xlabel(r'Requested probes ($k$)', fontsize=22, labelpad=9)
            ax.set_xticks([1, 2, 4, 6, 8, 10] if row_layout else range(1, 11))
            ax.set_xlim(0.7, 10.3)
            if metric == 'medae':
                ax.set_ylim(4, 10)
                ax.set_yticks(range(4, 11))
            else:
                ax.set_ylim(7, 18)
                ax.set_yticks(range(8, 19, 2))
            ax.tick_params(axis='both', labelsize=18)
            ax.grid(axis='y', color=GRAY, linewidth=0.75, alpha=0.22)
            if row_layout:
                ax.set_ylabel('MedAE (score points)' if metric == 'medae' else 'MedAPE (%)',
                              fontsize=21, labelpad=9)
            plotted.extend({
                'ordering': ordering, 'requested_probes': r['k'],
                'test_cells': r['n_cells_in_prediction_payload'],
                **({'metric': metric} if len(metrics) == 2 else {}),
                **{name: r[name][metric] for name, *_ in series},
            } for r in rows)
        metric_label = 'MedAE (score points)' if metric == 'medae' else 'MedAPE (%)'
        if not row_layout:
            axes[row_index, 0].set_ylabel(metric_label, fontsize=22, labelpad=12)
    save_fig(stem)
    figures = os.path.join(HERE, 'figures')
    with open(args.input, 'rb') as handle:
        source_hash = hashlib.sha256(handle.read()).hexdigest()
    write_json_atomic(os.path.join(figures, stem + '_data.json'), {
        'source': os.path.relpath(args.input, HERE),
        'source_sha256': source_hash,
        'source_manifest': data['manifest'],
        'metric': ('pooled MedAE on unrevealed observed cells' if args.metric == 'medae'
                   else 'pooled MedAPE on unrevealed observed cells with nonzero true score'
                   if args.metric == 'medape' else
                   'pooled MedAE and MedAPE on unrevealed observed cells; MedAPE excludes zero true scores'),
        'n_selection_models': 59, 'n_heldout_models': 25,
        'data': plotted,
    }, indent=2)
    caption = CAPTION
    if args.metric == 'medape':
        caption = caption.replace('median absolute error (MedAE,',
                                  'median absolute percentage error (MedAPE,')
        caption = caption.rstrip() + (
            ' Percentage errors use absolute error divided by the absolute true '
            'score, multiplied by 100; cells with zero true scores are excluded. '
            'Probe orderings remain those selected using MedAE.\n')
    if args.metric == 'both':
        caption = (
            '**Baselines on the 59/25 model split.** Probes are selected on 59 '
            'models; all 25 held-out models are evaluated at each budget. Panels '
            '(a,b) show median absolute error (MedAE); (c,d) show median absolute '
            'percentage error (MedAPE), both pooled over unrevealed scores '
            '(lower is better). The methods use identical revealed scores and '
            'test cells within each panel at each budget. Requested probes may '
            'be unavailable, and newly revealed scores leave the test set. '
            'Offsets only removes the two factors; model mean uses logit space.\n')
    with open(os.path.join(figures, stem + '_caption.md'), 'w') as handle:
        handle.write(caption)


if __name__ == '__main__':
    main()
