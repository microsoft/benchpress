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
    args = parser.parse_args()
    data = load_json(args.input)
    if data['config']['protocol'] != 'heldout_baselines_v2':
        raise ValueError('Expected the original held-out baseline protocol')
    panels = (
        ('cost_unaware_greedy', '(a) Cost-unaware greedy', PROBE_COST_UNAWARE_STYLE),
        ('cost_aware_greedy', '(b) Cost-aware greedy', PROBE_COST_AWARE_STYLE),
    )
    apply_single()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.75, wspace=0.13)
    plotted = []
    for ax, (ordering, title, policy_style) in zip(axes, panels):
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
            ax.plot([r['k'] for r in rows], [r[name]['medae'] for r in rows],
                    label=label, color=color, linestyle=linestyle, marker=marker,
                    linewidth=2.6, markersize=7.2, markeredgewidth=1.1,
                    markeredgecolor='white', zorder=4 if name == 'benchpress' else 3)
        ax.set_title(title, fontsize=23, pad=43, color=CHARCOAL)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.025), ncol=3,
                  frameon=False, fontsize=15.5, handlelength=1.55,
                  handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
        ax.set_xlabel(r'Requested probes ($k$)', fontsize=22, labelpad=9)
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.7, 10.3)
        ax.set_ylim(4, 10)
        ax.set_yticks(range(4, 11))
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(axis='y', color=GRAY, linewidth=0.75, alpha=0.22)
        plotted.extend({
            'ordering': ordering, 'requested_probes': r['k'],
            'test_cells': r['n_cells_in_prediction_payload'],
            **{name: r[name]['medae'] for name, *_ in series},
        } for r in rows)
    axes[0].set_ylabel('MedAE (score points)', fontsize=22, labelpad=12)
    save_fig(STEM)
    figures = os.path.join(HERE, 'figures')
    with open(args.input, 'rb') as handle:
        source_hash = hashlib.sha256(handle.read()).hexdigest()
    write_json_atomic(os.path.join(figures, STEM + '_data.json'), {
        'source': os.path.relpath(args.input, HERE),
        'source_sha256': source_hash,
        'source_manifest': data['manifest'],
        'metric': 'pooled MedAE on unrevealed observed cells',
        'n_selection_models': 59, 'n_heldout_models': 25,
        'data': plotted,
    }, indent=2)
    with open(os.path.join(figures, STEM + '_caption.md'), 'w') as handle:
        handle.write(CAPTION)


if __name__ == '__main__':
    main()
