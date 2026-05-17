#!/usr/bin/env python3
"""Plot: bp_matrix_clean_white"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import sys, os, json, csv, warnings
warnings.filterwarnings('ignore')

from benchpress.plot_helpers import style as S, data as D
from benchpress.plot_helpers.data import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_REASONING, MODEL_PROVIDERS, BENCH_CATS,
    MODEL_IDX, BENCH_IDX,
)
from benchpress.plot_helpers.style import *

# OBSERVED / M_FULL are already filtered to the paper-canonical (M≥15, B≥8)
# subset by `benchpress.build_benchmark_matrix` (single source of truth). This script
# just visualizes what the harness loads.


def fig_matrix_clean():
    print("[2] bp_matrix_clean_white")
    S.apply_single()
    sub = OBSERVED
    n_models_kept, n_bench_kept = sub.shape
    n_obs = int(sub.sum())
    fill = 100 * n_obs / (n_models_kept * n_bench_kept)
    print(f"    matrix: {n_models_kept} models \u00d7 {n_bench_kept} bench, "
          f"{n_obs} obs ({fill:.1f}% fill)")
    model_ord = np.argsort(-sub.sum(axis=1))
    bench_ord = np.argsort(-sub.sum(axis=0))
    obs_sorted = sub[model_ord][:, bench_ord]
    n_rows, n_cols = obs_sorted.shape

    # Models on Y (rows), benchmarks on X (columns)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = mcolors.ListedColormap(['white', S.VANILLA_BLUE])
    ax.imshow(obs_sorted, cmap=cmap, interpolation='nearest', aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(f'{n_bench_kept} benchmarks')
    ax.set_ylabel(f'{n_models_kept} models')
    # Title removed — redundant with caption
    S.save_fig('bp_matrix_clean_white')

if __name__ == "__main__":
    fig_matrix_clean()
