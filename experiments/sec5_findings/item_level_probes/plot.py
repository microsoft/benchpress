#!/usr/bin/env python
"""Plot held-out MedAE against the question budget from results/summary.json."""

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
    ANSWER_VIOLET, CYAN_TEAL, GRAY, MEMENTO_MAGENTA, ROSE, VANILLA_BLUE,
    apply_single, save_fig,
)

STYLES = {
    'subtasks': ('BenchPress, whole subtasks', MEMENTO_MAGENTA, '-', 'o'),
    'questions': ('BenchPress, questions', ROSE, '-', 's'),
    'random_sampling': ('Random-Sampling', GRAY, '--', 'o'),
    'random_sampling_learn': ('Random-Sampling-Learn', VANILLA_BLUE, '-', 'D'),
    'aipw': ('AIPW', CYAN_TEAL, '-', '^'),
    'gp_irt': ('GP-IRT', ANSWER_VIOLET, '-', 'v'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', default=os.path.join(HERE, 'results', 'summary.json'))
    ap.add_argument('--metric', choices=['medae_all', 'medae_unrevealed'], default='medae_all')
    ap.add_argument('--name', default='bp_question_level_budgets')
    args = ap.parse_args()

    with open(args.summary) as f:
        table = json.load(f)['table']
    apply_single()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, (label, color, ls, marker) in STYLES.items():
        rows = table[method]
        budgets = sorted(int(b) for b in rows)
        mean = np.array([rows[str(b)][f'{args.metric}_mean'] for b in budgets])
        std = np.array([rows[str(b)][f'{args.metric}_std'] for b in budgets])
        ax.plot(budgets, mean, color=color, linestyle=ls, marker=marker, label=label)
        if std.any():
            ax.fill_between(budgets, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax.set_xscale('log', base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel('Evaluated questions per model')
    ax.set_ylabel('MedAE (accuracy points)')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize='small')
    save_fig(args.name if args.metric == 'medae_all' else f'{args.name}_unrevealed')


if __name__ == '__main__':
    main()
