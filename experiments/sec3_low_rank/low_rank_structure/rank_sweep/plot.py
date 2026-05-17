#!/usr/bin/env python3
"""Plot: bp_rank_ucurve_raw_logit — raw/logit Soft-Impute SVD rank sweep.

Reads results.json (from run.py) and produces bp_rank_ucurve_raw_logit.pdf.
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, ROOT)

from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET, MEMENTO_MAGENTA, VANILLA_BLUE, apply_single, save_fig,
)
from benchpress.io_utils import load_json


METHODS = [
    ('identity_svd', 'Raw space', VANILLA_BLUE, 'o'),
    ('logit_svd', 'Logit space', MEMENTO_MAGENTA, 's'),
]


def _load_series(data, key, ranks):
    """Load per-rank metrics, tolerant of old/new key names."""
    return {
        'medape': [data[key][str(r)]['medape'] for r in ranks],
        'medae':  [data[key][str(r)]['medae']  for r in ranks],
    }


def main():
    data = load_json(os.path.join(HERE, 'results.json'))

    ranks = data['ranks']

    apply_single()
    fig, ax = plt.subplots(figsize=(4.4, 1.95), constrained_layout=True)

    all_vals = []
    for key, label, color, marker in METHODS:
        if key not in data:
            continue
        series = _load_series(data, key, ranks)
        vals = series['medape']
        all_vals.extend(vals)
        best_idx = int(np.argmin(vals))
        best_rank = ranks[best_idx]
        best_val = vals[best_idx]

        ax.plot(ranks, vals, marker=marker, linestyle='-', color=color,
                linewidth=2.4, markersize=6, label=label, zorder=2)
        ax.scatter([best_rank], [best_val], marker='*', s=180, color=ANSWER_VIOLET,
                   edgecolor=color, linewidth=0.8, zorder=4)

    ax.set_xlabel('Rank', fontsize=13)
    ax.set_ylabel('MedAPE (%)', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xticks(ranks)
    ax.set_xlim(min(ranks) - 0.5, max(ranks) + 0.5)
    ax.set_ylim(min(all_vals) - 0.35, max(all_vals) + 0.35)
    ax.legend(loc='best', frameon=False, fontsize=11)

    save_fig('bp_rank_ucurve_raw_logit')
    print("Done.")


if __name__ == '__main__':
    main()
