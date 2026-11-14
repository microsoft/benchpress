#!/usr/bin/env python
"""Plot saved hidden-question metrics against budget, one panel per selection."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.plot_helpers.visual_identity import (  # noqa: E402
    ANSWER_VIOLET, CYAN_TEAL, DEEP_BLUE, GRAY, LAVENDER, MEMENTO_MAGENTA,
    ROSE, VANILLA_BLUE, apply_single, save_fig,
)

STYLES = {
    'benchpress_r2': ('BenchPress (rank 2)', ROSE, '-', 's'),
    'benchpress_r4': ('BenchPress (rank 4)', ROSE, '--', 'P'),
    'benchpress_r8': ('BenchPress (rank 8)', ROSE, ':', 'X'),
    'benchpress_r40': ('BenchPress (rank 40)', ROSE, '-', 'P'),
    'random_sampling': ('Random-Sampling', GRAY, '--', 'o'),
    'random_sampling_learn': ('Random-Sampling-Learn', VANILLA_BLUE, '-', 'D'),
    'aipw': ('AIPW', CYAN_TEAL, '-', '^'),
    'gp_irt': ('GP-IRT', ANSWER_VIOLET, '-', 'v'),
    'variant_logit': ('BenchPress (logit ALS)', ROSE, '-', 'P'),
    'variant_raw': ('Raw ALS (rank 8)', MEMENTO_MAGENTA, '-', 's'),
    'variant_group_logit': ('Logit ALS + subtask offsets', LAVENDER, '--', 'v'),
    'variant_group_raw': ('Raw ALS + subtask offsets', CYAN_TEAL, '-.', 'P'),
    'variant_group_binary': ('Bernoulli + subtask offsets', DEEP_BLUE, ':', 'X'),
    'variant_raw_shrink': ('Raw ALS + shrunk residuals', MEMENTO_MAGENTA, '-.', '^'),
    'variant_task_mean_ridge': ('Subtask-mean ridge', DEEP_BLUE, '-', 'o'),
    'variant_task_item_pcr': ('Subtask ridge + item PCs', MEMENTO_MAGENTA, '-', '*'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', default=os.path.join(HERE, 'results', 'summary.json'))
    ap.add_argument('--name', default='bp_question_level_budgets')
    ap.add_argument('--methods', nargs='+', choices=list(STYLES))
    args = ap.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    table = summary['table']
    metric = summary.get('config', {}).get('metric', {})
    accuracy = metric.get('name') == 'binary_accuracy'
    methods = args.methods if args.methods is not None else [
        method for method in STYLES if method in table['random']]
    if accuracy and any(set(methods) != set(rows) for rows in table.values()):
        raise ValueError('the accuracy figure must include every scored method')
    field = 'accuracy' if accuracy else 'medae'
    has_variants = any(method.startswith('variant_') for method in methods)
    titles = {'random': 'Random questions', 'anchors': 'GP-IRT anchor questions'}
    apply_single()
    height = 8 if accuracy else (7 if has_variants else 5.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, height), sharey=True)
    for ax, (selection, title) in zip(axes, titles.items()):
        for method in methods:
            label, color, ls, marker = STYLES[method]
            if method == 'variant_logit':
                label = f'BenchPress (rank {summary["config"]["rank"]})'
            elif method == 'variant_raw' and 'rank' in summary.get('config', {}):
                label = f'Raw ALS (rank {summary["config"]["rank"]})'
            if accuracy and method in metric['subtask_constant_methods']:
                label += r' $\dagger$'
            rows = table[selection][method]
            budgets = sorted(int(b) for b in rows)
            mean = np.array([rows[str(b)][field + '_mean'] for b in budgets])
            std = np.array([rows[str(b)][field + '_std'] for b in budgets])
            ax.plot(budgets, mean, color=color, linestyle=ls, marker=marker, label=label)
            ax.fill_between(budgets, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xscale('log', base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel('Evaluated questions per model')
        if accuracy:
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Accuracy (%)' if accuracy else 'MedAE (accuracy points)')
    if has_variants:
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.5, 0.025 if accuracy else -0.02),
                   ncol=3, frameon=False, fontsize=11)
        fig.subplots_adjust(bottom=0.32 if accuracy else 0.25)
    else:
        axes[0].legend(frameon=False, fontsize='small')
    if accuracy:
        fig.text(0.5, 0.01,
                 r'$\dagger$ Subtask-constant predictions. Hidden questions only; '
                 r'predict 1 iff score $>0.5$. Bands: one seed standard deviation.',
                 ha='center', fontsize=10)
    save_fig(args.name)


if __name__ == '__main__':
    main()
