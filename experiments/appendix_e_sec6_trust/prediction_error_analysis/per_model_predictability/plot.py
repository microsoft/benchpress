#!/usr/bin/env python3
"""
Plot per-model predictability under the BenchPress predictor.

Horizontal bar chart: each bar = one model, sorted by MedAPE,
colored by provider. Vertical threshold at 15%.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.plot_helpers.visual_identity import (
    PROVIDER_COLORS, GRAY, MEMENTO_MAGENTA, apply_tall, save_fig,
)
from benchpress.io_utils import load_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot():
    results_path = os.path.join(SCRIPT_DIR, 'results.json')
    data = load_json(results_path)

    rows = data['results']  # sorted by MedAPE ascending
    n = len(rows)
    mid = (n + 1) // 2  # left panel gets one more if odd

    names = [r['model_name'] for r in rows]
    medapes = [r['medape'] for r in rows]
    providers = [r['provider'] for r in rows]
    colors = [PROVIDER_COLORS.get(p, GRAY) for p in providers]

    apply_tall()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(5.5, 0.26 * mid + 0.55),
                                      sharey=False)

    for ax, start, end in [(ax_l, 0, mid), (ax_r, mid, n)]:
        chunk_names = names[start:end]
        chunk_medapes = medapes[start:end]
        chunk_colors = colors[start:end]
        nk = len(chunk_names)

        y_pos = np.arange(nk)
        ax.barh(y_pos, chunk_medapes, color=chunk_colors, edgecolor='white',
                linewidth=0.3, height=0.78)

        ax.axvline(15, color=MEMENTO_MAGENTA, ls='--', lw=1.2, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(chunk_names, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_ylim(nk - 0.5, -0.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel('MedAPE (%)', fontsize=9)

        local_max = max(chunk_medapes)
        ax.set_xlim(0, local_max * 1.15)

    # Provider legend — single row across the top
    import matplotlib.patches as mpatches
    seen = {}
    for p in providers:
        if p not in seen:
            seen[p] = PROVIDER_COLORS.get(p, GRAY)
    handles = [mpatches.Patch(color=v, label=k) for k, v in seen.items()]
    fig.legend(handles=handles, loc='upper center', fontsize=8,
               ncol=6, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.06), handlelength=1.2,
               handletextpad=0.4, columnspacing=1.0)

    fig.subplots_adjust(left=0.30, right=0.98, top=0.90, bottom=0.05,
                        wspace=1.4)
    save_fig('bp_model_predictability')


if __name__ == '__main__':
    plot()
