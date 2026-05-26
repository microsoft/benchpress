#!/usr/bin/env python
"""Plot hard-rule temporal-deployment error distributions."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.artifact_utils import ensure_artifacts
from benchpress.io_utils import load_json
from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET,
    CHARCOAL,
    GRAY,
    MEMENTO_MAGENTA,
    VANILLA_BLUE,
    save_fig,
)

RESULTS_PATH = os.path.join(HERE, "results.json")
DISPLAY_K = [1, 5, 10]
EXPECTED_PROTOCOL = "temporal_deployment_hard_rule_v4"
K_COLORS = [VANILLA_BLUE, MEMENTO_MAGENTA, ANSWER_VIOLET]


def _apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _metric_values(payload: dict, metric: str) -> list[list[float]]:
    values = []
    for k in DISPLAY_K:
        k_values = []
        for landmark in payload["landmarks"]:
            summary = payload["summary_by_family"][landmark["family_key"]]
            value = summary["by_k"][str(k)][metric]["median"]
            if value is not None and np.isfinite(float(value)):
                k_values.append(float(value))
        values.append(k_values)
    return values


def _plot_metric(ax, values: list[list[float]], ylabel: str):
    positions = np.arange(len(DISPLAY_K))
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": CHARCOAL, "linewidth": 1.4},
        boxprops={"edgecolor": CHARCOAL, "linewidth": 0.9},
        whiskerprops={"color": CHARCOAL, "linewidth": 0.8},
        capprops={"color": CHARCOAL, "linewidth": 0.8},
    )
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(K_COLORS[idx])
        patch.set_alpha(0.16)

    rng = np.random.RandomState(42)
    for idx, ys in enumerate(values):
        jitter = rng.uniform(-0.11, 0.11, size=len(ys))
        ax.scatter(
            np.full(len(ys), positions[idx]) + jitter,
            ys,
            s=14,
            color=K_COLORS[idx],
            alpha=0.62,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        if ys:
            med = float(np.median(ys))
            ax.text(
                positions[idx],
                med,
                f"{med:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=CHARCOAL,
                fontweight="bold",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in DISPLAY_K])
    ax.set_ylabel(ylabel, labelpad=2)
    ax.grid(axis="y", color=GRAY, alpha=0.25, linewidth=0.8)


def main():
    ensure_artifacts(
        [RESULTS_PATH],
        ["{python}", os.path.join(HERE, "run.py"), "--mode", "merge"],
        description="Section 5.3 temporal-deployment hard-rule results",
    )
    payload = load_json(RESULTS_PATH)
    protocol = payload.get("config", {}).get("protocol_version")
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f"{RESULTS_PATH} has protocol {protocol!r}; expected {EXPECTED_PROTOCOL!r}. "
            "Run `python run.py --mode run-all` and `python run.py --mode merge` first."
        )
    n_targets = len(payload["landmarks"])

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.05), sharex=True)
    _plot_metric(axes[0], _metric_values(payload, "medae"), "MedAE")
    _plot_metric(axes[1], _metric_values(payload, "medape"), "MedAPE (%)")
    for ax in axes:
        ax.set_xlabel(r"Seed scores $k$", labelpad=1)
    fig.tight_layout(w_pad=1.4, pad=0.25)
    save_fig("bp_temporal_deployment_boxplot")


if __name__ == "__main__":
    main()
