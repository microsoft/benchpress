#!/usr/bin/env python
"""Plot saved hidden-question metrics against budget, one panel per selection."""

import argparse
import json
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.plot_helpers.visual_identity import (  # noqa: E402
    ANSWER_VIOLET, CYAN_TEAL, DEEP_BLUE, GRAY, LAVENDER, MEMENTO_MAGENTA,
    PROBE_RANDOM_STYLE, ROSE, VANILLA_BLUE, apply_single, save_fig,
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
    'gp_irt_items': ('IRT predictor (GP-IRT)', ANSWER_VIOLET, '-', 'v'),
    'aipw_items': ('Ridge predictor (AIPW)', CYAN_TEAL, '-', '^'),
}


def plot_selection(summary, name, caption=None):
    table = summary['table']
    ranks = summary['config']['ranks']
    policies = {
        'random': dict(PROBE_RANDOM_STYLE, label='Random questions'),
        'benchpress_anchors': {
            'label': 'BenchPress-selected questions', 'color': MEMENTO_MAGENTA,
            'linestyle': '-', 'marker': 's',
        },
        'anchors': {
            'label': 'GP-IRT-selected questions', 'color': ANSWER_VIOLET,
            'linestyle': ':', 'marker': 'v',
        },
    }
    present = set(next(iter(table.values()), {}))
    if (not summary['complete_grid']
            or summary['config']['metric']['name'] != 'binary_accuracy'
            or not ranks or set(table) != {f'benchpress_r{rank}' for rank in ranks}
            or not {'random', 'benchpress_anchors'}.issubset(present)
            or not present.issubset(policies)
            or any(set(rows) != present for rows in table.values())):
        raise ValueError('selection figure requires the same supported policies at every rank')
    policies = {policy: style for policy, style in policies.items() if policy in present}
    apply_single()
    fig, axes = plt.subplots(1, len(ranks), figsize=(7 * len(ranks), 6.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, rank in zip(axes, ranks):
        for policy, style in policies.items():
            rows = table[f'benchpress_r{rank}'][policy]
            budgets = sorted(map(int, rows))
            mean = np.array([rows[str(b)]['accuracy_mean'] for b in budgets])
            std = np.array([rows[str(b)]['accuracy_std'] for b in budgets])
            ax.plot(budgets, mean, **style)
            ax.fill_between(budgets, mean - std, mean + std,
                            color=style['color'], alpha=0.15, linewidth=0)
        ax.set_title(f'BenchPress predictor: rank {rank}')
        ax.set_xscale('log', base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel('Question budget per model' if 'anchors' in present
                      else 'Evaluated questions per model')
        ax.grid(alpha=0.3)
    lower, upper = axes[0].get_ylim()
    axes[0].set_ylim(max(0, lower), min(100, upper))
    axes[0].set_ylabel('Accuracy (%)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(0.5, 0.05 if 'anchors' in present else 0.07),
               ncol=2, frameon=False, fontsize=14)
    fig.subplots_adjust(bottom=0.28)
    default_caption = (
        '127 held-out models; identical unrevealed questions within each budget/seed.'
        '\nPredict 1 iff score >0.5. Mean and one sample SD over five selection seeds.')
    if 'anchors' in present:
        default_caption = (
            '127 held-out models; common hidden questions within each budget/seed.'
            '\nFive-seed mean ± sample SD; score >0.5 predicts 1. GP-IRT can use fewer unique questions.')
    if caption is None:
        fig.text(0.5, 0.01, default_caption, ha='center', fontsize=10)
    else:
        fig.text(0.5, 0.01, textwrap.fill(caption, width=90 * len(ranks),
                                       break_long_words=False, break_on_hyphens=False),
                 ha='center', va='top', fontsize=12, linespacing=1.25)
    save_fig(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', default=os.path.join(HERE, 'results', 'summary.json'))
    ap.add_argument('--name')
    ap.add_argument('--methods', nargs='+', choices=list(STYLES))
    ap.add_argument('--selections', nargs='+', choices=['random', 'anchors'])
    ap.add_argument('--xlabel')
    ap.add_argument('--caption', help='Replace the embedded note; wrap below the legend in both exports')
    args = ap.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    if summary.get('config', {}).get('comparison') == 'question_selection':
        if any(value is not None for value in (args.methods, args.selections, args.xlabel)):
            raise ValueError('the selection comparison uses all compared policies and its fixed axis labels')
        plot_selection(summary, args.name or 'bp_question_selection', caption=args.caption)
        return
    table = summary['table']
    metric = summary.get('config', {}).get('metric', {})
    accuracy = metric.get('name') == 'binary_accuracy'
    name = args.name or ('bp_question_accuracy' if accuracy else 'bp_question_level_budgets')
    methods = args.methods if args.methods is not None else [
        method for method in STYLES if method in table['random']]
    titles = {'random': 'Random questions', 'anchors': 'GP-IRT anchor questions'}
    selections = args.selections if args.selections is not None else list(titles)
    if len(set(methods)) != len(methods) or len(set(selections)) != len(selections):
        raise ValueError('methods and selections must be distinct')
    if accuracy and any(not set(methods).issubset(table[selection]) for selection in selections):
        raise ValueError('requested accuracy methods must be present in every selected panel')
    if accuracy and (not methods or not set(methods).issubset({
            'benchpress_r8', 'benchpress_r40', 'gp_irt_items', 'aipw_items'})):
        raise ValueError('accuracy requires native question predictors, not broadcast subtask means')
    field = 'accuracy' if accuracy else 'medae'
    has_variants = any(method.startswith('variant_') for method in methods)
    apply_single()
    height = 6.5 if accuracy else (7 if has_variants else 5.5)
    fig, axes = plt.subplots(1, len(selections), figsize=(7 * len(selections), height),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, selection in zip(axes, selections):
        for method in methods:
            label, color, ls, marker = STYLES[method]
            if method == 'variant_logit':
                label = f'BenchPress (rank {summary["config"]["rank"]})'
            elif method == 'variant_raw' and 'rank' in summary.get('config', {}):
                label = f'Raw ALS (rank {summary["config"]["rank"]})'
            rows = table[selection][method]
            budgets = sorted(int(b) for b in rows)
            mean = np.array([rows[str(b)][field + '_mean'] for b in budgets])
            std = np.array([rows[str(b)][field + '_std'] for b in budgets])
            zorder = 3 if accuracy and method in {
                'benchpress_r8', 'variant_group_logit'} else 2
            ax.plot(budgets, mean, color=color, linestyle=ls, marker=marker,
                    label=label, zorder=zorder)
            ax.fill_between(budgets, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_title(titles[selection])
        ax.set_xscale('log', base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel(args.xlabel if args.xlabel is not None else 'Evaluated questions per model')
        if not accuracy:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    if accuracy:
        lower, upper = axes[0].get_ylim()
        axes[0].set_ylim(max(0, lower), min(100, upper))
    axes[0].set_ylabel('Accuracy (%)' if accuracy else 'MedAE (accuracy points)')
    if accuracy:
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.5, 0.04 if len(selections) == 1 else 0.06),
                   ncol=2, frameon=False, fontsize=15)
        fig.subplots_adjust(bottom=0.28)
    elif has_variants:
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.5, 0.025 if accuracy else -0.02),
                   ncol=3, frameon=False, fontsize=11)
        fig.subplots_adjust(bottom=0.32 if accuracy else 0.25)
    else:
        axes[0].legend(frameon=False, fontsize='small')
    if args.caption is not None:
        fig.text(0.5, 0.01, textwrap.fill(args.caption, width=90 * len(selections),
                                       break_long_words=False, break_on_hyphens=False),
                 ha='center', va='top', fontsize=12, linespacing=1.25)
    elif accuracy:
        fig.text(0.5, 0.01,
                 r'Hidden questions only; predict 1 iff score $>0.5$. Bands: one seed standard deviation.'
                 '\nBaselines use their internal question predictors, not their aggregate score estimators.',
                 ha='center', fontsize=10)
    save_fig(name)


if __name__ == '__main__':
    main()
