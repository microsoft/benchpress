#!/usr/bin/env python
"""Plot MedAE on unevaluated questions against the question budget, one panel per selection."""

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
    'random_sampling': ('Random-Sampling', GRAY, '--', 'o'),
    'random_sampling_learn': ('Random-Sampling-Learn', VANILLA_BLUE, '-', 'D'),
    'aipw': ('AIPW', CYAN_TEAL, '-', '^'),
    'gp_irt': ('GP-IRT', ANSWER_VIOLET, '-', 'v'),
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
        table = json.load(f)['table']
    methods = args.methods if args.methods is not None else [
        method for method in STYLES if method in table['random']]
    has_variants = any(method.startswith('variant_') for method in methods)
    titles = {'random': 'Random questions', 'anchors': 'GP-IRT anchor questions'}
    apply_single()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7 if has_variants else 5.5), sharey=True)
    for ax, (selection, title) in zip(axes, titles.items()):
        for method in methods:
            label, color, ls, marker = STYLES[method]
            rows = table[selection][method]
            budgets = sorted(int(b) for b in rows)
            mean = np.array([rows[str(b)]['medae_mean'] for b in budgets])
            std = np.array([rows[str(b)]['medae_std'] for b in budgets])
            ax.plot(budgets, mean, color=color, linestyle=ls, marker=marker, label=label)
            ax.fill_between(budgets, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xscale('log', base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel('Evaluated questions per model')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('MedAE (accuracy points)')
    if has_variants:
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=11)
        fig.subplots_adjust(bottom=0.25)
    else:
        axes[0].legend(frameon=False, fontsize='small')
    save_fig(args.name)


if __name__ == '__main__':
    main()
