#!/usr/bin/env python3
"""Plot the matrix-completer informed-vs-blind prompt ablation."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys
sys.path.insert(0, os.path.join(HERE := os.path.dirname(os.path.abspath(__file__)), ".."))
from shared import benchmark_mean_median_metric, median_metric, protocol_matches
from benchpress.io_utils import load_json

from benchpress.plot_helpers.visual_identity import (
    apply_single,
    save_fig,
    CHARCOAL,
    GRAY,
    SOL_BASE01,
    LLM_MODEL_NAMES,
    LLM_MODEL_COLORS,
)

MODEL_ORDER = ["gpt-5.5"]

def plot():
    data = load_json(os.path.join(HERE, "results.json"))
    if not protocol_matches(data.get("protocol", {})):
        raise ValueError(
            "results.json was produced by an old LLM protocol; rerun run.py "
            "on the method-comparison folds before plotting."
        )

    apply_single()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    models = [
        m for m in MODEL_ORDER
        if m in data.get("informed", {}) or m in data.get("blind", {})
    ]
    inf_vals, bld_vals = [], []
    for model in models:
        inf_vals.append(median_metric(data.get("informed", {}).get(model, [])))
        bld_vals.append(median_metric(data.get("blind", {}).get(model, [])))

    bp_med = median_metric(data.get("bp", []))
    mean_med = benchmark_mean_median_metric()

    y = np.arange(len(models))
    height = 0.22
    colors = [LLM_MODEL_COLORS.get(m, GRAY) for m in models]

    ax.barh(
        y - height / 2,
        inf_vals,
        height,
        color=colors,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.8,
        label="Matrix completer: informed",
    )
    ax.barh(
        y + height / 2,
        bld_vals,
        height,
        color=colors,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.8,
        label="Matrix completer: blind",
    )

    if not np.isnan(bp_med):
        ax.axvline(bp_med, color=CHARCOAL, linestyle="--", linewidth=1.3)
    if not np.isnan(mean_med):
        ax.axvline(mean_med, color=SOL_BASE01, linestyle=":", linewidth=1.3)

    xmax = max(inf_vals + bld_vals + [bp_med, mean_med]) * 1.22
    for ypos, value in zip(np.r_[y - height / 2, y + height / 2],
                           inf_vals + bld_vals):
        if np.isnan(value):
            continue
        ax.text(
            value + xmax * 0.012,
            ypos,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(["" for _ in models])
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.set_xlabel("MedAPE (%)")
    ax.set_xlim(left=0, right=xmax)
    ax.grid(axis="x", alpha=0.3)

    handles = [
        Patch(facecolor=GRAY, alpha=0.95, edgecolor="white", label="Matrix completer: informed"),
        Patch(facecolor=GRAY, alpha=0.35, edgecolor="white", label="Matrix completer: blind"),
        Line2D(
            [0], [0], color=CHARCOAL, ls="--", lw=1.3,
            label=f"BenchPress ({bp_med:.1f}%)" if not np.isnan(bp_med) else "BenchPress",
        ),
        Line2D(
            [0], [0], color=SOL_BASE01, ls=":", lw=1.3,
            label=f"Benchmark mean ({mean_med:.1f}%)" if not np.isnan(mean_med) else "Benchmark mean",
        ),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, framealpha=0.9, fontsize=11, handlelength=1.6,
              columnspacing=1.2, borderpad=0.4)

    save_fig("bp_llm_completer")


if __name__ == "__main__":
    plot()
