#!/usr/bin/env python3
"""Plot confidence-calibration diagnostics."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from benchpress.artifact_utils import ensure_artifacts
from benchpress.plot_helpers.visual_identity import (
    CHARCOAL, MEMENTO_MAGENTA, VANILLA_BLUE, save_fig
)
from benchpress.io_utils import load_json

# Reuse run.py conformal helpers without repeating the math.
import importlib.util
_RUN_SPEC = importlib.util.spec_from_file_location(
    "_conf_run", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"))
_RUN = importlib.util.module_from_spec(_RUN_SPEC)
_RUN_SPEC.loader.exec_module(_RUN)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")
SCORES_PATH = os.path.join(SCRIPT_DIR, "confidence_scores.npz")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
DISPLAY_METHODS = [
    "disagreement",
    "structural_support",
    "combined_risk_model",
]


def _label(name):
    return {
        "bias_als_hp_disagreement": "Bias ALS HP",
        "strong_method_disagreement": "Strong-method spread",
        "disagreement": "Ensemble-spread",
        "structural_support": "Matrix-support",
        "combined_risk_model": "Hybrid uncertainty",
    }.get(name, name.replace("_", " ").title())


def _load_methods():
    ensure_artifacts(
        [RESULTS_PATH, SCORES_PATH],
        ["{python}", os.path.join(SCRIPT_DIR, "run.py"), "--ensure"],
        description="Section 4.4 confidence-calibration artifacts",
    )
    results = load_json(RESULTS_PATH)

    methods = {
        name: results["confidence_methods"][name]
        for name in DISPLAY_METHODS
        if name in results["confidence_methods"]
    }
    if not methods:
        raise ValueError("No confidence methods found in results.json")
    return methods


def _apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })


def _plot_risk_coverage(ax, methods, colors, xlabel="Predictions kept (%)",
                        metric="medape", show_title=True):
    metric_label = {"medae": "MedAE", "medape": "MedAPE (%)"}[metric]
    title = {"medae": "Risk-coverage (MedAE)",
             "medape": "Risk-coverage (MedAPE)"}[metric]
    for idx, (name, payload) in enumerate(methods.items()):
        rows = payload["risk_coverage_curve"]
        x = [r["kept_fraction"] * 100 for r in rows]
        y = [r[metric] for r in rows]
        ax.plot(x, y, "o-", color=colors[idx % len(colors)],
                linewidth=2.0, markersize=5.5, label=_label(name))
    ax.invert_xaxis()
    ax.set_xlim(104, 4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_label)
    if show_title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4,
              fontsize=8, labelspacing=0.3, borderaxespad=0.4)


def _plot_interval_width(ax, methods, colors, value_fontsize=9,
                         metric="absolute"):
    if metric == "absolute":
        key, fmt, xlabel, title = (
            "median_width", "{:.2f}",
            "Median interval width",
            "Conformal 90% intervals",
        )
    elif metric == "relative":
        key, fmt, xlabel, title = (
            "median_relative_width", "{:.1f}%",
            "Median interval width (% of actual)",
            "Conformal 90% intervals (MedAPE scale)",
        )
    else:
        raise ValueError(f"unknown metric {metric!r}")
    labels, widths = [], []
    for name, payload in methods.items():
        interval = payload.get("conformal_90_interval", {})
        labels.append(_label(name))
        widths.append(interval.get(key, float("nan")))
    y = list(range(len(labels)))
    bar_colors = [colors[idx % len(colors)] for idx in range(len(labels))]
    ax.barh(y, widths, color=bar_colors, alpha=0.85, height=0.55)
    span = max(widths) - min(widths)
    pad = max(span * 0.03, 0.4)
    for idx, width in enumerate(widths):
        ax.text(width + pad, y[idx], fmt.format(width),
                va="center", ha="left", fontsize=value_fontsize,
                color=colors[idx % len(colors)])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(min(widths) - pad, max(widths) + pad * 6)
    ax.grid(axis="x", alpha=0.25)


def _compute_widths_multi_ci(npz_path, methods, ci_levels=(0.80, 0.90, 0.95)):
    """Recompute conformal interval median widths for each method at each CI level."""
    data = np.load(npz_path, allow_pickle=True)
    actual = data["actual"]
    predicted = data["predicted"]
    fold_id = data["fold_id"]
    uncertainty_keys = {
        "disagreement": "disagreement_uncertainty",
        "structural_support": "structural_support_uncertainty",
        "combined_risk_model": "combined_risk_model_uncertainty",
    }
    out = {}
    for name in methods:
        unc = data[uncertainty_keys[name]]
        out[name] = {}
        for ci in ci_levels:
            lo, hi, _ = _RUN._conformal_interval(actual, predicted, unc, fold_id, ci=ci)
            cw = _RUN._coverage_width(actual, lo, hi)
            out[name][ci] = {
                "coverage": cw["coverage"],
                "median_width": cw["median_width"],
            }
    return out


def _print_latex_table(widths, ci_levels=(0.80, 0.90, 0.95)):
    """Emit a LaTeX fragment for the conformal-width table to stdout."""
    print("% --- conformal interval widths table fragment ---")
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"\toprule")
    headers = " & ".join(f"{int(ci*100)}\\%" for ci in ci_levels)
    print(rf"Method & {headers} \\")
    print(r"\midrule")
    label_map = {
        "disagreement": "Ensemble spread",
        "structural_support": "Matrix support",
        "combined_risk_model": r"\textbf{Hybrid uncertainty}",
    }
    for name in widths:
        row_vals = " & ".join(f"{widths[name][ci]['median_width']:.2f}" for ci in ci_levels)
        if name == "combined_risk_model":
            row_vals = " & ".join(f"\\textbf{{{widths[name][ci]['median_width']:.2f}}}" for ci in ci_levels)
        print(f"{label_map[name]} & {row_vals} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("% --- end fragment ---")


def main():
    methods = _load_methods()
    _apply_style()
    colors = [CHARCOAL, VANILLA_BLUE, MEMENTO_MAGENTA]

    npz_path = os.path.join(SCRIPT_DIR, "confidence_scores.npz")
    widths = _compute_widths_multi_ci(npz_path, list(methods.keys()))
    _print_latex_table(widths)

    # Main figure: just the risk-coverage curve, no title (caption carries the
    # name). Sized short so it sits next to a small 4-row table without empty
    # whitespace.
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4))
    _plot_risk_coverage(ax, methods, colors,
                        xlabel="Predictions kept (%)",
                        metric="medae", show_title=False)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    save_fig("bp_confidence_calibration")
    print(os.path.join(FIG_DIR, "bp_confidence_calibration.pdf"))

    # NeurIPS still uses single-panel risk-coverage in its appendix.
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.1))
    _plot_risk_coverage(ax, methods, colors,
                        xlabel="Most confident predictions kept (%)",
                        metric="medae")
    fig.tight_layout()
    save_fig("bp_confidence_risk_coverage_neurips")
    print(os.path.join(FIG_DIR, "bp_confidence_risk_coverage_neurips.pdf"))

    # NeurIPS main body uses a compact bar of conformal interval widths.
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.0))
    _plot_interval_width(ax, methods, colors, value_fontsize=10,
                         metric="absolute")
    fig.tight_layout()
    save_fig("bp_confidence_interval_width_neurips")
    print(os.path.join(FIG_DIR, "bp_confidence_interval_width_neurips.pdf"))


if __name__ == "__main__":
    main()
