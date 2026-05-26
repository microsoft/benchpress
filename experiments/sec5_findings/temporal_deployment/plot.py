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
    CHARCOAL,
    GRAY,
    MEMENTO_MAGENTA,
    VANILLA_BLUE,
    apply_double,
    save_fig,
)

RESULTS_PATH = os.path.join(HERE, "results.json")
DISPLAY_K = [1, 5, 10]
EXPECTED_PROTOCOL = "temporal_deployment_hard_rule_v4"


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


def _plot_metric(ax, values: list[list[float]], ylabel: str, title: str):
    positions = np.arange(len(DISPLAY_K))
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": CHARCOAL, "linewidth": 2.0},
        boxprops={"edgecolor": CHARCOAL, "linewidth": 1.2},
        whiskerprops={"color": CHARCOAL, "linewidth": 1.0},
        capprops={"color": CHARCOAL, "linewidth": 1.0},
    )
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor([VANILLA_BLUE, MEMENTO_MAGENTA, VANILLA_BLUE][idx])
        patch.set_alpha(0.16)

    rng = np.random.RandomState(42)
    for idx, ys in enumerate(values):
        jitter = rng.uniform(-0.11, 0.11, size=len(ys))
        ax.scatter(
            np.full(len(ys), positions[idx]) + jitter,
            ys,
            s=28,
            color=MEMENTO_MAGENTA if idx == 1 else VANILLA_BLUE,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.45,
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
                fontsize=9,
                color=CHARCOAL,
                fontweight="bold",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in DISPLAY_K])
    ax.set_xlabel("Revealed seed scores per target model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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

    apply_double()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)
    _plot_metric(axes[0], _metric_values(payload, "medae"), "MedAE (points)", "Absolute error")
    _plot_metric(axes[1], _metric_values(payload, "medape"), "MedAPE (%)", "Percentage error")
    fig.suptitle(
        f"Temporal deployment across {n_targets} post-R1 target models",
        y=1.04,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig("bp_temporal_deployment_boxplot")


if __name__ == "__main__":
    main()
