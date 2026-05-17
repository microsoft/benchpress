#!/usr/bin/env python3
"""Shared axes helpers for BenchPress plots."""

import numpy as np

from benchpress.plot_helpers.visual_identity import MEMENTO_MAGENTA, VANILLA_BLUE


def style_dual_axes(
    ax,
    ax2,
    left_color=MEMENTO_MAGENTA,
    right_color=VANILLA_BLUE,
    grid_alpha=0.16,
):
    ax.tick_params(axis="y", colors=left_color, length=0, pad=3)
    ax2.tick_params(axis="y", colors=right_color, length=0, pad=3)
    ax.yaxis.label.set_color(left_color)
    ax2.yaxis.label.set_color(right_color)
    ax.spines["left"].set_color(left_color)
    ax2.spines["right"].set_color(right_color)
    ax.grid(True, axis="y", alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax._right_axis = ax2


def binned_medians(x, y, n_bins=5, log_x=False, quantile=False):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if quantile:
        bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    elif log_x:
        x_pos = x[x > 0]
        bins = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), n_bins + 1)
    else:
        bins = np.linspace(x.min(), x.max(), n_bins + 1)

    cx, cy = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (x >= lo) & (x <= hi) if i == n_bins - 1 else (x >= lo) & (x < hi)
        if m.sum() > 0:
            if quantile:
                cx.append(float(np.median(x[m])))
            elif log_x:
                cx.append(float(np.sqrt(lo * hi)))
            else:
                cx.append(float((lo + hi) / 2))
            cy.append(float(np.median(y[m])))
    return np.array(cx), np.array(cy)
