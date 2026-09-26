#!/usr/bin/env python
"""Plot hard-rule temporal-deployment error distributions."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.io_utils import load_json
from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET,
    CHARCOAL,
    GRAY,
    MEMENTO_MAGENTA,
    VANILLA_BLUE,
    save_fig,
)
from summary_utils import temporal_overall_medians, temporal_values_by_k

RESULTS_PATH = os.path.join(HERE, "results.json")
OVERLEAF_FIGURES_DIR = os.path.abspath(os.path.join(
    REPO_ROOT, "..", "overleaf", "iclr2027", "figures",
))
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


def _metric_values(payload: dict, metric: str, hidden_only: bool) -> list[list[float]]:
    if hidden_only:
        return temporal_values_by_k(payload, metric, hidden_only=True)
    return [
        [
            float(payload["summary_by_family"][landmark["family_key"]]["by_k"][str(k)][metric]["median"])
            for landmark in payload["landmarks"]
            if payload["summary_by_family"][landmark["family_key"]]["by_k"][str(k)][metric]["median"] is not None
        ]
        for k in DISPLAY_K
    ]


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


def _save_temporal_figure(hidden_only: bool, out_pdf: str | None):
    if hidden_only:
        out_pdf = out_pdf or os.path.join(
            OVERLEAF_FIGURES_DIR, "bp_temporal_deployment_boxplot.pdf",
        )
        os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        print(f"  -> {out_pdf}")
        return
    save_fig("bp_temporal_deployment_boxplot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-only", action="store_true",
                        help="Plot non-revealed cells only, using merged raw rows.")
    parser.add_argument("--out-pdf", default=None,
                        help="PDF output path. Hidden-only defaults to the ICLR Overleaf figures directory.")
    args = parser.parse_args()

    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing merged results JSON: {RESULTS_PATH}")
    payload = load_json(RESULTS_PATH)
    protocol = payload.get("config", {}).get("protocol_version")
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f"{RESULTS_PATH} has protocol {protocol!r}; expected {EXPECTED_PROTOCOL!r}. "
            "Run `python run.py --mode run-all` and `python run.py --mode merge` first."
        )
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.05), sharex=True)
    _plot_metric(axes[0], _metric_values(payload, "medae", args.hidden_only), "MedAE")
    _plot_metric(axes[1], _metric_values(payload, "medape", args.hidden_only), "MedAPE (%)")
    for ax in axes:
        ax.set_xlabel(r"Seed scores $k$", labelpad=1)
    fig.tight_layout(w_pad=1.4, pad=0.25)
    _save_temporal_figure(args.hidden_only, args.out_pdf)

    medians = temporal_overall_medians(payload, hidden_only=args.hidden_only)
    label = "hidden-only" if args.hidden_only else "with-probe-zero"
    print(f"{label} temporal medians:")
    for metric in ("medae", "medape"):
        print("  " + metric + " " + " ".join(
            f"k={k}:{medians[metric][k]:.2f}" for k in DISPLAY_K
        ))


if __name__ == "__main__":
    main()
