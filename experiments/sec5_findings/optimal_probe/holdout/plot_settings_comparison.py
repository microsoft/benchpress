"""Compare main and held-out probe experiments in one MedAE-only figure."""

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

STEM = 'bp_main_vs_heldout_medae'


def main():
    panels = (
        ('medae_any', 'cost_unaware_greedy', 'Cost-unaware greedy', PROBE_COST_UNAWARE_STYLE),
        ('medae_low_cost', 'cost_aware_greedy', 'Cost-aware greedy', PROBE_COST_AWARE_STYLE),
    )
    heldout_path = os.path.join(HERE, 'results', 'heldout_baselines_s42.json.gz')
    heldout = load_json(heldout_path)
    source_paths = [heldout_path]
    plotted = []
    apply_single()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.84,
                        wspace=0.13, hspace=0.64)
    fig.text(0.535, 0.978, 'Main experiment: 84 models', ha='center',
             va='top', fontsize=25, color=CHARCOAL)
    fig.text(0.535, 0.519, 'Held-out validation: 59 selection / 25 test models',
             ha='center', va='top', fontsize=25, color=CHARCOAL)
    for column, (main_order, heldout_order, title, style) in enumerate(panels):
        main_series = {}
        for method, suffix in (('benchpress', ''), ('logit_offsets_only', '_rank0')):
            path = os.path.join(HERE, '..', 'all_known', 'results',
                                f'fixed_order_{main_order}{suffix}_hidden_only.json.gz')
            source_paths.append(path)
            data = load_json(path)
            if data['config']['n_models'] != 84:
                raise ValueError('Main experiment must contain 84 models')
            main_series[method] = {row['k']: row for row in data['summary_by_k']}
        for row_index in range(2):
            ax = axes[row_index, column]
            if row_index == 0:
                rows = []
                for k in range(1, 11):
                    bp, offsets = (main_series[name][k] for name in
                                   ('benchpress', 'logit_offsets_only'))
                    if bp['probe_ids'] != offsets['probe_ids']:
                        raise ValueError('Main probe orderings differ across methods')
                    if bp['hidden_only']['n'] != offsets['hidden_only']['n']:
                        raise ValueError('Main prediction coverage differs across methods')
                    rows.append({'k': k, 'test_cells': bp['hidden_only']['n'],
                                 'benchpress': bp['hidden_only']['medae'],
                                 'logit_offsets_only': offsets['hidden_only']['medae']})
            else:
                source = sorted((r for r in heldout['summary']
                                 if r['method'] == heldout_order), key=lambda r: r['k'])
                rows = []
                for r in source:
                    names = ('benchpress', 'logit_offsets_only', 'logit_model_mean')
                    if any(r[name]['n'] != r['n_cells_in_prediction_payload'] for name in names):
                        raise ValueError('Held-out prediction coverage differs across methods')
                    rows.append({'k': r['k'], 'test_cells': r['n_cells_in_prediction_payload'],
                                 **{name: r[name]['medae'] for name in names}})
            if [r['k'] for r in rows] != list(range(1, 11)):
                raise ValueError('Expected all ten requested-probe budgets')
            series = [
                ('benchpress', 'BenchPress', style['color'], '-', 'o'),
                ('logit_offsets_only', 'Offsets only', ANSWER_VIOLET, '--', 's'),
            ]
            if row_index == 1:
                series.append(('logit_model_mean', 'Model mean', GRAY, ':', '^'))
            for name, label, color, linestyle, marker in series:
                ax.plot([r['k'] for r in rows], [r[name] for r in rows], label=label,
                        color=color, linestyle=linestyle, marker=marker, linewidth=2.6,
                        markersize=7.2, markeredgecolor='white', markeredgewidth=1.1,
                        zorder=4 if name == 'benchpress' else 3)
            letter = 'abcd'[row_index * 2 + column]
            ax.set_title(f'({letter}) {title}', fontsize=22, pad=38, color=CHARCOAL)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.015), ncol=len(series),
                      frameon=False, fontsize=15.5, handlelength=1.55,
                      handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
            ax.set_xticks(range(1, 11))
            ax.set_xlim(0.7, 10.3)
            ax.set_ylim(4, 10)
            ax.set_yticks(range(4, 11))
            ax.tick_params(axis='both', labelsize=18)
            ax.grid(axis='y', color=GRAY, linewidth=0.75, alpha=0.22)
            if column == 0:
                ax.set_ylabel('MedAE (score points)', fontsize=22, labelpad=12)
            if row_index == 1:
                ax.set_xlabel(r'Requested probes ($k$)', fontsize=22, labelpad=9)
            plotted.extend({'setting': 'main' if row_index == 0 else 'heldout',
                            'ordering': heldout_order, **r} for r in rows)
    save_fig(STEM)
    sources = []
    for path in source_paths:
        with open(path, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        sources.append({'path': os.path.relpath(path, HERE), 'sha256': digest})
    write_json_atomic(os.path.join(HERE, 'figures', STEM + '_data.json'), {
        'metric': 'pooled MedAE on unrevealed observed cells',
        'sources': sources, 'data': plotted,
    }, indent=2)
    caption = (
        '**Main and held-out probe experiments.** Top: probe selection and '
        'evaluation on the main 84-model matrix; each target is predicted using '
        'the other 83 model rows and its available probe scores. Bottom: probes '
        'selected on 59 models are evaluated on 25 held-out models, one target '
        'at a time. Left: cost-unaware greedy; right: cost-aware greedy. All '
        'panels show pooled median absolute error (MedAE, lower is better), '
        'using the original MedAE-selected probe orderings. Methods share the '
        'same revealed scores and test cells within each panel at each budget. '
        'The settings use different probe sets and evaluation cells. Requested '
        'probes may be unavailable, and newly revealed scores leave the test set. '
        'Offsets only removes the two latent factors. Model mean is shown for '
        'the held-out comparison.\n')
    with open(os.path.join(HERE, 'figures', STEM + '_caption.md'), 'w') as handle:
        handle.write(caption)


if __name__ == '__main__':
    main()
