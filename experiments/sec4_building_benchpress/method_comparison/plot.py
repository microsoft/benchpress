#!/usr/bin/env python3
"""Plot: bp_transform_method_grid — heatmaps of Section 4 metrics across transform × method combos.

Source: results.json derived from prediction shards by run.py --merge.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from benchpress.plot_helpers import style as S
from benchpress.io_utils import load_json

TRANSFORM_NAMES = ['Identity', 'Log', 'Logit', 'Arcsinh', 'Square root', 'Probit', 'Quantile']
TRANSFORM_KEYS  = ['identity', 'log', 'logit', 'asinh', 'sqrt', 'probit', 'quantile']
METHOD_NAMES = ['Benchmark Mean', 'Model Mean', 'Bench-KNN', 'Model-KNN',
                'BenchReg', 'ModelReg', 'Soft-Impute', 'Bias ALS',
                'NMF', 'PMF', 'Nuclear Norm', 'MLP']

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

# Theme colormap: magenta/violet (good) -> vanilla/cyan (high).
CMAP_LOWER_BETTER = mcolors.LinearSegmentedColormap.from_list(
    'bp_low', [S.MEMENTO_MAGENTA, S.ANSWER_VIOLET, S.VANILLA_BLUE, S.CYAN_TEAL], N=256
)
# Reversed: for metrics where higher is better (Coverage)
CMAP_HIGHER_BETTER = mcolors.LinearSegmentedColormap.from_list(
    'bp_high', [S.CYAN_TEAL, S.VANILLA_BLUE, S.ANSWER_VIOLET, S.MEMENTO_MAGENTA], N=256
)

ALL_METRICS = [
    ('medape_median',     r'MedAPE (%) $\downarrow$',  'lower'),
    ('medae_median',      r'MedAE $\downarrow$',       'lower'),
    ('coverage',          r'Coverage (%) $\uparrow$',  'higher'),
]



def load_grids(methods=None):
    """Load grid data. If methods is given, only include those methods."""
    if methods is None:
        methods = METHOD_NAMES
    results = load_json(RESULTS_PATH)
    grids = {}
    for key, label, direction in ALL_METRICS:
        grid = np.zeros((len(methods), len(TRANSFORM_NAMES)))
        for ti, tkey in enumerate(TRANSFORM_KEYS):
            for mi, mname in enumerate(methods):
                grids_val = results[tkey][mname].get(key, float('nan'))
                grid[mi, ti] = grids_val
        grids[key] = (grid, label, direction)
    return grids

def draw_heatmap(ax, grid, label, direction='lower', show_ylabel=True,
                 fontsize_cell=13, methods=None, top_k=0, highlight_best=False):
    """Draw a single heatmap on the given axes.

    Args:
        top_k: if > 0, draw a bold rectangle around the top-k best cells.
        highlight_best: if True, highlight only the single best cell.
        methods: list of method names for y-axis labels.
    """
    if methods is None:
        methods = METHOD_NAMES
    n_m, n_t = grid.shape
    cmap = CMAP_LOWER_BETTER if direction == 'lower' else CMAP_HIGHER_BETTER

    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    if vmax - vmin > 1:
        vmin, vmax = np.floor(vmin), np.ceil(vmax)

    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_t))
    ax.set_xticklabels(TRANSFORM_NAMES)
    ax.set_yticks(range(n_m))
    if show_ylabel:
        ax.set_yticklabels(methods)
    else:
        ax.set_yticklabels([])

    ax.set_title(label, fontweight='bold', pad=8)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(S.CHARCOAL)
        spine.set_linewidth(0.5)

    # Find cells to highlight — compare using displayed (formatted) values
    top_k_cells = set()
    effective_k = top_k
    if highlight_best:
        effective_k = 1
    if effective_k > 0:
        # Format values the same way they appear in the heatmap
        if 'Coverage' in label:
            disp_fn = lambda v: round(v * 100, 0)
        else:
            disp_fn = lambda v: round(v, 1)

        disp_vals = []
        for mi in range(n_m):
            for ti in range(n_t):
                v = grid[mi, ti]
                if not np.isnan(v):
                    disp_vals.append((disp_fn(v), mi, ti))

        if direction == 'lower':
            disp_vals.sort(key=lambda x: x[0])
        else:
            disp_vals.sort(key=lambda x: -x[0])

        if disp_vals:
            best_disp = disp_vals[0][0]
            for dv, mi, ti in disp_vals[:max(effective_k, len(disp_vals))]:
                if dv == best_disp:
                    top_k_cells.add((mi, ti))
                else:
                    break

    mid = (vmin + vmax) / 2
    if 'Coverage' in label:
        fmt_fn = lambda v: f'{v*100:.0f}'
    else:
        fmt_fn = lambda v: f'{v:.1f}'

    for mi in range(n_m):
        for ti in range(n_t):
            val = grid[mi, ti]
            if direction == 'lower':
                color = 'white' if val > mid else S.CHARCOAL
            else:
                color = 'white' if val < mid else S.CHARCOAL
            ax.text(ti, mi, fmt_fn(val), ha='center', va='center',
                    fontsize=fontsize_cell, color=color, fontweight='bold')

            if (mi, ti) in top_k_cells:
                rect = plt.Rectangle((ti - 0.5, mi - 0.5), 1, 1,
                                     linewidth=2.5, edgecolor=S.CHARCOAL,
                                     facecolor='none', zorder=10)
                ax.add_patch(rect)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.outline.set_linewidth(0.5)
    return im

def plot_summary_grid():
    """One-row layout with Section 4 metrics (all 12 methods — full version for appendix)."""
    S.apply_double()
    grids = load_grids()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    for idx, (key, label, direction) in enumerate(ALL_METRICS):
        ax = axes[idx]
        grid, _, _ = grids[key]
        draw_heatmap(ax, grid, label, direction=direction,
                     show_ylabel=(idx == 0), fontsize_cell=12)

    plt.tight_layout()
    S.save_fig('bp_transform_method_grid_scores')

def plot_single():
    """Single MedAPE heatmap (original)."""
    S.apply_double()
    grids = load_grids()
    grid, label, direction = grids['medape_median']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_heatmap(ax, grid, label, direction=direction)
    ax.set_xlabel('Feature transform')
    ax.set_ylabel('Prediction method')

    plt.tight_layout()
    S.save_fig('bp_transform_method_grid')

if __name__ == "__main__":
    plot_single()
    plot_summary_grid()
