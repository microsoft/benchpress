#!/usr/bin/env python3
"""Plot the five-shot predictor prompt ablation."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(HERE := os.path.dirname(os.path.abspath(__file__)), ".."))
from shared import median_metric, protocol_matches
from benchpress.io_utils import load_json

from benchpress.plot_helpers.visual_identity import (
    CHARCOAL,
    GRAY,
    SOL_BASE01,
    apply_single,
    save_fig,
)


def plot():
    path = os.path.join(HERE, "results.json")
    data = load_json(path)
    if not protocol_matches(data.get("protocol", {})):
        raise ValueError("results.json was produced by an old protocol; rerun run.py.")

    model = "gpt-5.5"
    named = median_metric(data.get("five_shot_named", {}).get(model, []))
    blind = median_metric(data.get("five_shot_blind", {}).get(model, []))
    bp_med = median_metric(data.get("bp", []))

    apply_single()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(2)
    vals = [named, blind]
    labels = ["Named", "Blind"]
    for xpos, value, alpha in zip(x, vals, [0.95, 0.35]):
        ax.bar(xpos, value, color=GRAY, alpha=alpha, edgecolor="white", linewidth=0.8)
    for xpos, value in zip(x, vals):
        if np.isfinite(value):
            ax.text(xpos, value + 0.35, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    if np.isfinite(bp_med):
        ax.axhline(bp_med, color=CHARCOAL, linestyle="--", linewidth=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MedAPE (%)")
    ymax = max([v for v in vals + [bp_med] if np.isfinite(v)] + [1.0]) * 1.22
    ax.set_ylim(bottom=0, top=ymax)
    ax.grid(axis="y", alpha=0.3)
    handles = [
        Patch(facecolor=GRAY, alpha=0.95, edgecolor="white", label="Five-shot named"),
        Patch(facecolor=GRAY, alpha=0.35, edgecolor="white", label="Five-shot blind"),
        Line2D([0], [0], color=CHARCOAL, ls="--", lw=1.3,
               label=f"BenchPress ({bp_med:.1f}%)" if np.isfinite(bp_med) else "BenchPress"),
        Line2D([0], [0], color=SOL_BASE01, ls=":", lw=1.3, label="Matrix-completer result: see companion plot"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    save_fig("bp_llm_five_shot_predictor")


if __name__ == "__main__":
    plot()
