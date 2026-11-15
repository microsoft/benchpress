"""Render the Appendix D.4 prediction and selection panels from saved summaries."""

import argparse
import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np

from benchpress.io_utils import load_json, write_json
from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET, MEMENTO_MAGENTA, PROBE_RANDOM_STYLE, apply_single, save_fig,
)
from experiments.sec5_findings.item_level_probes.plot import STYLES


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accuracy-summary')
    parser.add_argument('--selection-summary')
    args = parser.parse_args()
    if bool(args.accuracy_summary) != bool(args.selection_summary):
        parser.error('supply both source summaries to rebuild the figure data')
    result_path = os.path.join(HERE, 'results.json')
    prediction_methods = ['benchpress_r40', 'gp_irt_items', 'aipw_items']
    selection_styles = {
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
    if args.accuracy_summary:
        accuracy = load_json(args.accuracy_summary)
        selection = load_json(args.selection_summary)
        if not all(s['complete_grid'] and s['config']['metric']['name'] == 'binary_accuracy'
                   for s in (accuracy, selection)):
            raise ValueError('both sources must contain complete binary-accuracy results')
        results = {'sources': {}, 'prediction': {}, 'selection': {}}
        for name, path, source in (
                ('prediction', args.accuracy_summary, accuracy),
                ('selection', args.selection_summary, selection)):
            with open(path, 'rb') as stream:
                checksum = hashlib.sha256(stream.read()).hexdigest()
            results['sources'][name] = {
                'config_hash': source['config_hash'], 'summary_sha256': checksum,
                'derived_from': 'complete saved summary; table entries copied without recomputation',
            }
        for method in prediction_methods:
            results['prediction'][method] = accuracy['table']['random'][method]
        for policy in selection_styles:
            results['selection'][policy] = {
                str(b): selection['table']['benchpress_r40'][policy][str(b)]
                for b in (200, 400, 800)
            }
        write_json(result_path, results)
    results = load_json(result_path)
    apply_single()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, panel in zip(axes, ('prediction', 'selection')):
        names = prediction_methods if panel == 'prediction' else selection_styles
        for name in names:
            if panel == 'prediction':
                label, color, linestyle, marker = STYLES[name]
                style = dict(label=label, color=color, linestyle=linestyle, marker=marker)
            else:
                style = selection_styles[name]
            rows = results[panel][name]
            budgets = sorted(map(int, rows))
            mean = np.array([rows[str(b)]['accuracy_mean'] for b in budgets])
            std = np.array([rows[str(b)]['accuracy_std'] for b in budgets])
            ax.plot(budgets, mean, **style)
            ax.fill_between(budgets, mean - std, mean + std,
                            color=style['color'], alpha=0.15, linewidth=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(budgets, labels=[str(b) for b in budgets])
        ax.set_xlabel('Known scores per model' if panel == 'prediction'
                      else 'Question budget per model', fontsize=22)
        ax.set_ylabel('Accuracy (%)', fontsize=22)
        ax.tick_params(labelsize=20)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28),
                  frameon=False, fontsize=18, ncol=1)
    axes[0].set_title('(a) Question-level prediction', fontsize=23)
    axes[1].set_title('(b) Question selection', fontsize=23)
    axes[1].set_yticks([79.5, 80, 80.5, 81, 81.5])
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.35, wspace=0.27)
    save_fig('bp_question_prediction_selection')


if __name__ == '__main__':
    main()
