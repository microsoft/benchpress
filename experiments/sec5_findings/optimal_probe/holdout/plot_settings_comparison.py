"""Compare main and held-out probe experiments in one MedAE-only figure."""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.io_utils import load_json, write_json_atomic
from benchpress.stats import cluster_bootstrap_median
from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET, CHARCOAL, GRAY, PROBE_COST_AWARE_STYLE,
    PROBE_COST_UNAWARE_STYLE, apply_single, save_fig,
)
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

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
        main_cells = {}
        for method, suffix in (('benchpress', ''), ('logit_offsets_only', '_rank0')):
            path = os.path.join(HERE, '..', 'all_known', 'results',
                                f'fixed_order_{main_order}{suffix}_hidden_only.json.gz')
            source_paths.append(path)
            data = load_json(path)
            if data['config']['n_models'] != 84:
                raise ValueError('Main experiment must contain 84 models')
            main_series[method] = {row['k']: row for row in data['summary_by_k']}
            raw = [r for r in data['raw_predictions'] if not r['is_revealed']]
            main_cells[method] = {
                (r['k'], r['target_model_id'], r['benchmark_id']): r for r in raw
            }
            if len(main_cells[method]) != len(raw):
                raise ValueError('Duplicate main-experiment prediction cells')
        if main_cells['benchpress'].keys() != main_cells['logit_offsets_only'].keys():
            raise ValueError('Main methods must be bootstrapped on identical cells')
        for row_index in range(2):
            ax = axes[row_index, column]
            panel_records = {}
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
                    panel_records[k] = []
                    for key in sorted(main_cells['benchpress']):
                        if key[0] != k:
                            continue
                        bp_cell = main_cells['benchpress'][key]
                        offset_cell = main_cells['logit_offsets_only'][key]
                        if bp_cell['actual'] != offset_cell['actual']:
                            raise ValueError('Main methods have different test labels')
                        panel_records[k].append({
                            'model_id': key[1], 'true': bp_cell['actual'],
                            'benchpress': bp_cell['pred'],
                            'logit_offsets_only': offset_cell['pred'],
                        })
            else:
                source = sorted((r for r in heldout['summary']
                                 if r['method'] == heldout_order), key=lambda r: r['k'])
                rows = []
                for r in source:
                    names = ('benchpress', 'logit_offsets_only')
                    if any(r[name]['n'] != r['n_cells_in_prediction_payload'] for name in names):
                        raise ValueError('Held-out prediction coverage differs across methods')
                    rows.append({'k': r['k'], 'test_cells': r['n_cells_in_prediction_payload'],
                                 **{name: r[name]['medae'] for name in names}})
                    panel_records[r['k']] = [
                        {'model_id': cell['model_id'], 'true': cell['true'],
                         **{name: cell[name + '_pred'] for name in names}}
                        for cell in heldout['raw_cells']
                        if cell['method'] == heldout_order and cell['k'] == r['k']
                    ]
            if [r['k'] for r in rows] != list(range(1, 11)):
                raise ValueError('Expected all ten requested-probe budgets')
            series = [
                ('benchpress', 'BenchPress', style['color'], '-', 'o'),
                ('logit_offsets_only', 'Offsets only', ANSWER_VIOLET, '--', 's'),
            ]
            for row in rows:
                records = panel_records[row['k']]
                if len(records) != row['test_cells']:
                    raise ValueError('Raw prediction count does not match the plotted statistic')
                clusters = [cell['model_id'] for cell in records]
                expected_models = 84 if row_index == 0 else 25
                if len(set(clusters)) != expected_models:
                    raise ValueError('Unexpected model cohort in bootstrap')
                row['bootstrap'] = {}
                for name, *_ in series:
                    errors = [abs(cell[name] - cell['true']) for cell in records]
                    interval = cluster_bootstrap_median(errors, clusters)
                    if not np.isclose(interval['median'], row[name], atol=1e-10, rtol=0):
                        raise ValueError('Bootstrap estimand differs from the reported pooled MedAE')
                    row['bootstrap'][name] = interval
            for name, label, color, linestyle, marker in series:
                lower = np.array([r['bootstrap'][name]['ci_lower'] for r in rows])
                upper = np.array([r['bootstrap'][name]['ci_upper'] for r in rows])
                ax.errorbar([r['k'] for r in rows], (lower + upper) / 2,
                            yerr=(upper - lower) / 2, fmt='none',
                            ecolor=to_rgba(color, 0.55), elinewidth=1.1,
                            capsize=3, capthick=1.1, zorder=2)
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
    intervals = [interval for r in plotted for interval in r['bootstrap'].values()]
    lower_limit = min(4, np.floor(min(r['ci_lower'] for r in intervals)))
    upper_limit = max(10, np.ceil(max(r['ci_upper'] for r in intervals)))
    axes[0, 0].set_ylim(lower_limit, upper_limit)
    axes[0, 0].set_yticks(np.arange(lower_limit, upper_limit + 1))
    save_fig(STEM)
    sources = []
    for path in source_paths:
        with open(path, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        sources.append({'path': os.path.relpath(path, HERE), 'sha256': digest})
    write_json_atomic(os.path.join(HERE, 'figures', STEM + '_data.json'), {
        'metric': 'pooled MedAE on unrevealed observed cells',
        'uncertainty': {
            'method': 'pointwise percentile model-cluster bootstrap',
            'confidence': 0.95, 'n_resamples': 2000, 'seed': 42,
            'paired_draws': 'same model resamples across methods and budgets within each setting',
            'fixed': 'model split, selected probe orderings, and fitted predictions',
        },
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
        'All four panels compare BenchPress with offsets only, which removes '
        'the two latent factors. Bars are pointwise 95% percentile intervals '
        'from 2,000 model-cluster bootstrap resamples (seed 42), paired across '
        'methods and budgets within each setting. Model splits, probe '
        'orderings, and fitted predictions remain fixed.\n')
    with open(os.path.join(HERE, 'figures', STEM + '_caption.md'), 'w') as handle:
        handle.write(caption)


if __name__ == '__main__':
    main()
