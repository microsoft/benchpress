#!/usr/bin/env python3
"""Plot bp_rank_ucurve_holdout_models: rank sweep with the newest 20% of models
held out as a block, raw and logit space, with 95% cluster-bootstrap intervals.

Reads rank_sweep_holdout_models.json (from holdout_models.py) and produces
bp_rank_ucurve_holdout_models.pdf for app:rank_geometry.
"""
import os, sys
import numpy as np
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


def main():
    data = load_json(os.path.join(HERE, 'rank_sweep_holdout_models.json'))
    summary = data['protocols']['model']['summary']

    apply_single()
    fig, ax = plt.subplots(figsize=(4.6, 2.15), constrained_layout=True)

    all_vals = []
    for key, label, color, marker in METHODS:
        rows = summary[key]['rows']
        ranks = [r['rank'] for r in rows]
        med = [r['medape'] for r in rows]
        lo = [r['medape_ci'][0] for r in rows]
        hi = [r['medape_ci'][1] for r in rows]
        all_vals += lo + hi
        ax.fill_between(ranks, lo, hi, color=color, alpha=0.15, zorder=1)
        ax.plot(ranks, med, marker=marker, linestyle='-', color=color,
                linewidth=2.3, markersize=5.5, label=label, zorder=2)
        best = int(np.argmin(med))
        ax.scatter([ranks[best]], [med[best]], marker='*', s=175,
                   color=ANSWER_VIOLET, edgecolor=color, linewidth=0.8, zorder=4)

    ax.set_xlabel('Rank', fontsize=13)
    ax.set_ylabel('Held-out MedAPE (%)', fontsize=13)
    ax.set_xticks(ranks)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(min(all_vals) - 0.4, max(all_vals) + 0.4)
    ax.legend(loc='upper center', frameon=False, fontsize=11, ncol=2)

    save_fig('bp_rank_ucurve_holdout_models')
    print("Done.")


if __name__ == '__main__':
    main()
