#!/usr/bin/env python3
"""Appendix expansion of \\Cref{fig:error_hypotheses_52}.

Main-text §4.3 figure shows only four hypotheses (H1, H5, H8, H9), single
metric per panel. This appendix variant shows the **full grid**:
all 9 hypotheses (H1..H9) x both score-error metrics (MedAPE, MedAE).
To keep labels readable in the paper, the figure is split into two horizontal
blocks: H1--H5 on the left and H6--H9 on the right.

Reads results.json from sec4_building_benchpress/error_analysis/model_analysis/H{1..9}_*/.
Output: figures/bp_model_predictability_factors_full.{pdf,png}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET, CYAN_TEAL, MEMENTO_MAGENTA, VANILLA_BLUE,
    CHARCOAL, apply_double, save_fig,
)
from benchpress.io_utils import load_json

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "sec4_building_benchpress",
                   "error_analysis", "model_analysis")


def _load(subdir):
    return load_json(os.path.join(SRC, subdir, "results.json"))


# --- score-error metrics, in column order ---------------------------------
METRICS = [
    {"key": "medape",     "col_title": r"MedAPE (%) $\downarrow$",        "fmt": "{:.1f}"},
    {"key": "medae",      "col_title": r"MedAE (pts) $\downarrow$",       "fmt": "{:.2f}"},
]

# --- correlational hypotheses (H1..H4) -----------------------------------
# H2 is binary (reasoning vs not) -> grouped bars.
CORR_HYPS = [
    {"row": 0, "id": "H1", "dir": "H1_model_size",          "feature": "log_params",
     "xlabel": r"$\log_{10}$(params)",      "log_x": False, "kind": "scatter"},
    {"row": 1, "id": "H2", "dir": "H2_model_type",          "feature": "is_reasoning",
     "kind": "binary"},
    {"row": 2, "id": "H3", "dir": "H3_score_level",         "feature": "med_true",
     "xlabel": "Median observed\nscore (pts)", "log_x": False, "kind": "scatter"},
    {"row": 3, "id": "H4", "dir": "H4_rank2_expressibility","feature": "rank2_R2",
     "xlabel": r"Row $R^2$ (rank-2)",       "log_x": False, "kind": "scatter"},
]

# --- ablation hypotheses (H5..H8) ----------------------------------------
ABL_HYPS = [
    {"row": 4, "id": "H5", "dir": "H5_neighbor_quality",  "kind": "single_bar",
     "treat_label": "W/o strong\npeers",        "base_label": "W/ strong\npeers"},
    {"row": 5, "id": "H6", "dir": "H6_neighbor_evidence", "kind": "single_bar",
     "treat_label": "Mask 75%\npeer overlap",  "base_label": "Full peer\noverlap"},
    {"row": 6, "id": "H7", "dir": "H7_family_peers",      "kind": "single_bar",
     "treat_label": "W/o same-\nprovider peers", "base_label": "W/ same-\nprovider peers"},
    {"row": 7, "id": "H8", "dir": "H8_observation_count", "kind": "drop_line",
     "fracs": [25, 50, 75], "xlabel": "Target obs.\nhidden (%)"},
]

# H9 row: temporal A vs B across k.
H9_HYP = {"row": 8, "id": "H9", "dir": "H9_temporal", "kind": "temporal",
          "ks": [1, 3, 5, 10], "xlabel": "Revealed benchmarks $k$"}


# --------------------------------------------------------------------------
# Panel drawers
# --------------------------------------------------------------------------

def _scatter_binned(ax, x, y, n_bins=5, log_x=False):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        ax.text(0.5, 0.5, "n<3 finite", ha="center", va="center",
                transform=ax.transAxes, color=CHARCOAL, fontsize=16)
        return
    ax.scatter(x, y, c=CYAN_TEAL, s=30, alpha=0.45, zorder=2)
    if log_x and (x > 0).sum() >= n_bins + 1:
        bins = np.logspace(np.log10(x[x > 0].min()), np.log10(x[x > 0].max()), n_bins + 1)
    else:
        bins = np.linspace(x.min(), x.max(), n_bins + 1)
        log_x = False
    cx, cy = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (x >= lo) & (x <= hi) if i == n_bins - 1 else (x >= lo) & (x < hi)
        if m.sum() > 0:
            cx.append(np.sqrt(lo * hi) if log_x else (lo + hi) / 2)
            cy.append(np.median(y[m]))
    ax.plot(cx, cy, "o-", color=MEMENTO_MAGENTA, lw=3.0, markersize=8, zorder=3)
    if log_x:
        ax.set_xscale("log")


def _binary_bars(ax, vals_a, vals_b, label_a, label_b, fmt):
    ma, mb = np.nanmedian(vals_a), np.nanmedian(vals_b)
    ax.bar([0, 1], [ma, mb], 0.55,
           color=[VANILLA_BLUE, MEMENTO_MAGENTA], alpha=0.9, edgecolor="white", linewidth=1.0,
           )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b], fontsize=15)
    for xi, v in zip([0, 1], [ma, mb]):
        ax.annotate(fmt.format(v), (xi, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=16, color=CHARCOAL)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.22)


def _two_bar(ax, base_v, treat_v, base_label, treat_label, fmt):
    ax.bar([0, 1], [base_v, treat_v], 0.55,
           color=[VANILLA_BLUE, MEMENTO_MAGENTA], alpha=0.9,
           edgecolor="white", linewidth=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([base_label, treat_label], fontsize=15)
    for xi, v in zip([0, 1], [base_v, treat_v]):
        ax.annotate(fmt.format(v), (xi, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=16, color=CHARCOAL)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.22)


def _line(ax, xs, vals, fmt, xlabel):
    ax.plot(xs, vals, "o-", color=MEMENTO_MAGENTA, lw=3.0, markersize=9, zorder=3)
    span = (max(vals) - min(vals)) or 1.0
    pad = span * 0.08
    ax.fill_between(xs, [v - pad for v in vals], [v + pad for v in vals],
                    color=MEMENTO_MAGENTA, alpha=0.07)
    for i, (x, v) in enumerate(zip(xs, vals)):
        xoff = 6 if i == 0 else 0
        ha = "left" if i == 0 else "center"
        ax.annotate(fmt.format(v), (x, v), textcoords="offset points",
                     xytext=(xoff, 10), ha=ha, fontsize=16, color=MEMENTO_MAGENTA)
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel, fontsize=15)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.22)


def _two_line(ax, xs, vals_a, vals_b, label_a, label_b, fmt, xlabel):
    ax.plot(xs, vals_a, "o-", color=ANSWER_VIOLET, lw=3.0, markersize=9, label=label_a, zorder=3)
    ax.plot(xs, vals_b, "s-", color=CYAN_TEAL,     lw=3.0, markersize=9, label=label_b, zorder=3)
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.legend(frameon=False, loc="best", fontsize=13)
    all_v = vals_a + vals_b
    span = (max(all_v) - min(all_v)) or 1.0
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.18)


# --------------------------------------------------------------------------
# Row drawers
# --------------------------------------------------------------------------

def draw_corr_scatter_row(axes_row, hyp):
    rows = _load(hyp["dir"])["models"]
    x = np.array([r[hyp["feature"]] for r in rows], dtype=float)
    for col, m in enumerate(METRICS):
        y = np.array([r.get(m["key"], np.nan) for r in rows], dtype=float)
        _scatter_binned(axes_row[col], x, y, log_x=hyp.get("log_x", False))
        axes_row[col].set_xlabel(hyp["xlabel"], fontsize=15)


def draw_binary_row(axes_row, hyp):
    rows = _load(hyp["dir"])["models"]
    nr = [r for r in rows if r["is_reasoning"] == 0]
    rr = [r for r in rows if r["is_reasoning"] == 1]
    for col, m in enumerate(METRICS):
        a = np.array([r.get(m["key"], np.nan) for r in nr], dtype=float)
        b = np.array([r.get(m["key"], np.nan) for r in rr], dtype=float)
        _binary_bars(axes_row[col], a, b,
                     f"Non-reas.\n(n={len(nr)})", f"Reasoning\n(n={len(rr)})",
                     m["fmt"])


def _baseline_per_model(metric_key):
    """Return median of per-model values from H1 results (baseline LOO)."""
    rows = _load("H1_model_size")["models"]
    vals = [r.get(metric_key, np.nan) for r in rows]
    return float(np.nanmedian(vals))


def draw_ablation_bar_row(axes_row, hyp):
    j = _load(hyp["dir"])
    tests = j.get("tests", {})
    for col, m in enumerate(METRICS):
        base_v = _baseline_per_model(m["key"])
        d = tests.get(m["key"], {}).get("median_delta", 0.0)
        treat_v = base_v + d
        _two_bar(axes_row[col], base_v, treat_v,
                 hyp["base_label"], hyp["treat_label"], m["fmt"])


def draw_drop_line_row(axes_row, hyp):
    """H8: hide_25pct, baseline (50%), hide_75pct."""
    j = _load(hyp["dir"])
    fracs = hyp["fracs"]
    for col, m in enumerate(METRICS):
        base = _baseline_per_model(m["key"])
        d25 = j["hide_25pct"]["tests"].get(m["key"], {}).get("median_delta", 0.0)
        d75 = j["hide_75pct"]["tests"].get(m["key"], {}).get("median_delta", 0.0)
        vals = [base + d25, base, base + d75]
        _line(axes_row[col], fracs, vals, m["fmt"], hyp["xlabel"])


def draw_h9_row(axes_row, hyp):
    j = _load(hyp["dir"])
    comp = j["comparison_A_vs_B"]
    ks = hyp["ks"]
    for col, m in enumerate(METRICS):
        a_vals = [comp[str(k)][m["key"]]["median_A"] for k in ks]
        b_vals = [comp[str(k)][m["key"]]["median_B"] for k in ks]
        _two_line(axes_row[col], ks, a_vals, b_vals,
                  "Oldest", "Middle", m["fmt"], hyp["xlabel"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    apply_double()
    plt.rcParams.update({
        "font.size": 19,
        "axes.titlesize": 23,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })

    fig = plt.figure(figsize=(19.4, 14.1))
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1.12, 1.0],
        left=0.015,
        right=0.99,
        top=0.91,
        bottom=0.07,
        wspace=0.11,
    )
    left_grid = outer[0].subgridspec(
        5, 3, width_ratios=[0.56, 1.0, 1.0], hspace=0.64, wspace=0.36
    )
    right_grid = outer[1].subgridspec(
        4, 3, width_ratios=[0.58, 1.0, 1.0], hspace=0.68, wspace=0.36
    )
    left_label_axes = [fig.add_subplot(left_grid[r, 0]) for r in range(5)]
    right_label_axes = [fig.add_subplot(right_grid[r, 0]) for r in range(4)]
    for ax in left_label_axes + right_label_axes:
        ax.axis("off")
    left_axes = np.array([
        [fig.add_subplot(left_grid[r, c + 1]) for c in range(2)]
        for r in range(5)
    ])
    right_axes = np.array([
        [fig.add_subplot(right_grid[r, c + 1]) for c in range(2)]
        for r in range(4)
    ])

    axes_by_hyp = {
        "H1": left_axes[0],
        "H2": left_axes[1],
        "H3": left_axes[2],
        "H4": left_axes[3],
        "H5": left_axes[4],
        "H6": right_axes[0],
        "H7": right_axes[1],
        "H8": right_axes[2],
        "H9": right_axes[3],
    }
    label_axes_by_hyp = {
        "H1": left_label_axes[0],
        "H2": left_label_axes[1],
        "H3": left_label_axes[2],
        "H4": left_label_axes[3],
        "H5": left_label_axes[4],
        "H6": right_label_axes[0],
        "H7": right_label_axes[1],
        "H8": right_label_axes[2],
        "H9": right_label_axes[3],
    }

    for hyp in CORR_HYPS:
        if hyp["kind"] == "scatter":
            draw_corr_scatter_row(axes_by_hyp[hyp["id"]], hyp)
        elif hyp["kind"] == "binary":
            draw_binary_row(axes_by_hyp[hyp["id"]], hyp)

    for hyp in ABL_HYPS:
        if hyp["kind"] == "single_bar":
            draw_ablation_bar_row(axes_by_hyp[hyp["id"]], hyp)
        elif hyp["kind"] == "drop_line":
            draw_drop_line_row(axes_by_hyp[hyp["id"]], hyp)

    draw_h9_row(axes_by_hyp["H9"], H9_HYP)

    for ax_block in (left_axes, right_axes):
        for col, m in enumerate(METRICS):
            ax_block[0, col].set_title(
                m["col_title"], fontsize=22, fontweight="bold", pad=12,
                color=CHARCOAL,
            )

    row_labels = {
        "H1": r"$H_1$ Model" "\nsize",
        "H2": r"$H_2$ Model" "\ntype",
        "H3": r"$H_3$ Score" "\nlevel",
        "H4": r"$H_4$ Low-rank" "\nfit",
        "H5": r"$H_5$ Strong-peer" "\npresence",
        "H6": r"$H_6$ Strong-peer" "\nsupport",
        "H7": r"$H_7$ Same-provider" "\nevidence",
        "H8": r"$H_8$ Observation" "\ncount",
        "H9": r"$H_9$ Training-anchor" "\nrecency",
    }
    for hyp_id, lab in row_labels.items():
        label_axes_by_hyp[hyp_id].text(
            0.98, 0.5, lab, transform=label_axes_by_hyp[hyp_id].transAxes,
            fontsize=17, fontweight="bold", ha="right", va="center",
            color=CHARCOAL, linespacing=1.05,
        )

    for ax in list(left_axes.flat) + list(right_axes.flat):
        ax.tick_params(axis="both", labelsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for x, title in [
        (0.28, "H1--H5"),
        (0.75, "H6--H9"),
    ]:
        fig.text(
            x, 0.965, title, ha="center", va="center", fontsize=25,
            fontweight="bold", color=CHARCOAL,
        )

    save_fig("bp_model_predictability_factors_full")
    print("Saved bp_model_predictability_factors_full.{pdf,png}")


if __name__ == "__main__":
    main()
