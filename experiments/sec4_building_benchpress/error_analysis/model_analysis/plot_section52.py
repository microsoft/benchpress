#!/usr/bin/env python3
"""Composite §4.3 model-side error-analysis figure.

Main-text panel selection emphasizes the strongest and most interpretable
current signals: H2, H3, H5, H8, and H9.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET, CYAN_TEAL, MEMENTO_MAGENTA, VANILLA_BLUE,
    apply_double, save_fig,
)
from benchpress.plot_helpers.axes import binned_medians, style_dual_axes
from benchpress.io_utils import load_json as load_json_file

HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(subdir):
    return load_json_file(os.path.join(HERE, subdir, "results.json"))


def _dual_scatter_binned(ax, x, medape, medae, xlabel, title):
    ax2 = ax.twinx()
    ax.scatter(x, medape, c=MEMENTO_MAGENTA, s=44, alpha=0.34, zorder=2)
    ax2.scatter(x, medae, c=VANILLA_BLUE, s=44, alpha=0.34, zorder=2)
    for axis, vals, color, marker in [
        (ax, medape, MEMENTO_MAGENTA, "o"),
        (ax2, medae, VANILLA_BLUE, "s"),
    ]:
        cx, cy = binned_medians(x, vals)
        axis.plot(cx, cy, marker + "-", color=color, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MedAPE (%)")
    ax2.set_ylabel("MedAE (pts)")
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    style_dual_axes(ax, ax2)


def _dual_line(ax, x, medape, medae, xlabel, title, xticklabels=None):
    ax2 = ax.twinx()
    ax.plot(x, medape, "o-", color=MEMENTO_MAGENTA, zorder=3)
    ax2.plot(x, medae, "s--", color=VANILLA_BLUE, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MedAPE (%)")
    ax2.set_ylabel("MedAE (pts)")
    ax.set_xticks(x)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    style_dual_axes(ax, ax2)


def _metric_pair_bar(ax, labels, metric_values, title):
    x = np.arange(2)
    w = 0.34
    colors = [VANILLA_BLUE, MEMENTO_MAGENTA]
    for j, (label, vals) in enumerate(zip(labels, metric_values)):
        ax.bar(x + (j - 0.5) * w, vals, width=w, color=colors[j],
               edgecolor="white", linewidth=0.7, alpha=0.92, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["MedAPE", "MedAE"])
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    ax.legend(frameon=False, loc="upper right", handlelength=0.9,
              handletextpad=0.30, borderaxespad=0.0, labelspacing=0.18,
              fontsize=16)
    ax.grid(True, axis="y", alpha=0.16)


def panel_h2(ax):
    """H2: Model type — grouped metric bars."""
    models = load_json("H2_model_type")["models"]
    r = [m for m in models if m["is_reasoning"] == 1]
    nr = [m for m in models if m["is_reasoning"] == 0]

    nr_medape = np.median([m["medape"] for m in nr])
    r_medape = np.median([m["medape"] for m in r])
    nr_medae = np.median([m["medae"] for m in nr])
    r_medae = np.median([m["medae"] for m in r])

    _metric_pair_bar(
        ax,
        ["Non-reas.", "Reasoning"],
        [[nr_medape, nr_medae], [r_medape, r_medae]],
        "$H_2$: Reasoning models\nare easier to predict",
    )


def panel_h3(ax):
    """H3: Score level — dual-axis scatter + binned trends."""
    models = load_json("H3_score_level")["models"]
    x = np.array([m["med_true"] for m in models])
    medape = np.array([m["medape"] for m in models])
    medae = np.array([m["medae"] for m in models])
    _dual_scatter_binned(
        ax, x, medape, medae, "Median score",
        "$H_3$: Higher-scoring models\nare easier to predict",
    )


def panel_h5(ax):
    data = load_json("H5_neighbor_quality")
    tests = data.get("tests", {})
    h1_models = load_json("H1_model_size")["models"]
    base_medape = np.median([m["medape"] for m in h1_models])
    base_medae = np.median([m["medae"] for m in h1_models])
    no_peer_medape = base_medape + tests.get("medape", {}).get("median_delta", 0)
    no_peer_medae = base_medae + tests.get("medae", {}).get("median_delta", 0)
    _metric_pair_bar(
        ax,
        ["Removed", "Present"],
        [[no_peer_medape, no_peer_medae], [base_medape, base_medae]],
        "$H_5$: Strong-peer presence\nreduces error",
    )


def panel_h8(ax):
    """H8: Observation count — dual-axis line across 3 hide fractions."""
    h1_models = load_json("H1_model_size")["models"]
    base_medape = np.median([m["medape"] for m in h1_models])
    base_medae = np.median([m["medae"] for m in h1_models])

    h8 = load_json("H8_observation_count")
    d25 = h8["hide_25pct"]["tests"]["medape"]["median_delta"]
    d75 = h8["hide_75pct"]["tests"]["medape"]["median_delta"]
    d25_medae = h8["hide_25pct"]["tests"]["medae"]["median_delta"]
    d75_medae = h8["hide_75pct"]["tests"]["medae"]["median_delta"]

    fracs = [25, 50, 75]
    medape = [base_medape + d25, base_medape, base_medape + d75]
    medae = [base_medae + d25_medae, base_medae, base_medae + d75_medae]
    _dual_line(
        ax, fracs, medape, medae, "Hidden (%)",
        "$H_8$: More observations\nreduce error",
    )


def panel_h9(ax):
    """H9: Temporal — dual-axis line for temporal gap."""
    h9 = load_json("H9_temporal")
    comp = h9["comparison_A_vs_B"]
    k_plot = [1, 3, 5, 10]

    medape_gap = [comp[str(k)]["medape"]["median_diff"] for k in k_plot]
    medae_gap = [comp[str(k)]["medae"]["median_diff"] for k in k_plot]
    _dual_line(
        ax, k_plot, medape_gap, medae_gap, "# revealed",
        "$H_9$: Training-anchor recency\nreduces error",
    )


def main():
    apply_double()
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "axes.titlepad": 5,
        "axes.labelpad": 4,
        "xtick.labelsize": 19,
        "ytick.labelsize": 17,
        "legend.fontsize": 18,
        "lines.linewidth": 3.2,
        "lines.markersize": 10,
    })
    fig, axes = plt.subplots(1, 5, figsize=(19.8, 4.5))

    panel_h2(axes[0])
    panel_h3(axes[1])
    panel_h5(axes[2])
    panel_h8(axes[3])
    panel_h9(axes[4])

    for ax in axes:
        ax.grid(True, axis="x", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.050, right=0.985, top=0.78, bottom=0.22,
                        wspace=0.58)
    save_fig("bp_error_hypotheses_52")


if __name__ == "__main__":
    main()
