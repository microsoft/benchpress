#!/usr/bin/env python3
"""Appendix expansion of \\Cref{fig:predictability_factors_51}.

Main-text figure shows the jointly supported benchmark-side patterns
(H3, H4, H5), one panel each. This appendix variant shows the full grid:
all 7 active hypotheses (H1..H7) x both score-error metrics (MedAPE, MedAE).
To keep labels readable in the paper, the figure is split into two horizontal
blocks: H1--H3 on the left and H4--H7 on the right.

Reads canonical results.json (corr) / ablation_results.json (abl) from
sec4_building_benchpress/error_analysis/benchmark_analysis/H*/.
Output: figures/bp_predictability_factors_full.{pdf,png}
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
                   "error_analysis", "benchmark_analysis")


def _load_corr(subdir):
    return load_json(os.path.join(SRC, subdir, "results.json"))


def _load_abl(subdir):
    return load_json(os.path.join(SRC, subdir, "ablation_results.json"))


# --- score-error metrics, in column order ----------------------------------
METRICS = [
    {"key": "medape",   "label": r"MedAPE (%) $\downarrow$",                   "col_title": r"MedAPE (%) $\downarrow$",        "fmt": "{:.1f}"},
    {"key": "medae",    "label": r"MedAE (pts) $\downarrow$",                  "col_title": r"MedAE (pts) $\downarrow$",       "fmt": "{:.2f}"},
]

# --- correlational hypotheses (H1..H3): feature vs metric ------------------
CORR_HYPS = [
    {"row": 0, "id": "H1", "dir": "H1_low_rank_fit",         "feature": "rank2_R2",
     "xlabel": r"Rank-2 $R^2$",                 "log_x": False, "title_arrow": "Higher low-rank fit"},
    {"row": 1, "id": "H2", "dir": "H2_score_level",          "feature": "med_score",
     "xlabel": "Median score\n(pts)",  "log_x": False, "title_arrow": "Higher score level"},
    {"row": 2, "id": "H3", "dir": "H3_score_spread",         "feature": "std_score",
     "xlabel": "Score std. dev.\n(log scale)",  "log_x": True,  "title_arrow": "Higher score spread"},
]

# --- ablation hypotheses (H4..H7): line / grouped-bar ----------------------
ABL_HYPS = [
    {"row": 3, "id": "H4", "dir": "H4_target_coverage",          "kind": "drop_line",
     "drs": [0.25, 0.5, 0.75], "xlabel": "Target obs.\ndropped (%)",
     "title": "Drop target coverage"},
    {"row": 4, "id": "H5", "dir": "H5_strong_neighbor_presence", "kind": "threshold_bar",
     "thr": 0.85,
     "treat_label": "W/o strong\nneighbors", "base_label": "W/ strong\nneighbors",
     "title": "Remove strong neighbors"},
    {"row": 5, "id": "H6", "dir": "H6_strong_neighbor_support",  "kind": "drop_line",
     "drs": [0.25, 0.5, 0.75], "xlabel": "Best-neighbor obs.\ndropped (%)",
     "title": "Drop best-neighbor obs."},
    {"row": 6, "id": "H7", "dir": "H7_same_category_evidence",   "kind": "single_bar",
     "treat_label": "W/o same-cat.\npeers", "base_label": "W/ same-cat.\npeers",
     "title": "Remove same-category peers"},
]


# ---------------------------------------------------------------------------
# Panel drawers
# ---------------------------------------------------------------------------

def _scatter_binned(ax, x, y, n_bins=5, log_x=False):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        ax.text(0.5, 0.5, "n<3 finite", ha="center", va="center",
                transform=ax.transAxes, color=CHARCOAL, fontsize=16)
        return
    ax.scatter(x, y, c=CYAN_TEAL, s=30, alpha=0.45, zorder=2)
    if log_x:
        x_pos = x[x > 0]
        if len(x_pos) >= n_bins + 1:
            bins = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), n_bins + 1)
        else:
            bins = np.linspace(x.min(), x.max(), n_bins + 1); log_x = False
    else:
        bins = np.linspace(x.min(), x.max(), n_bins + 1)
    cx, cy = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (x >= lo) & (x <= hi) if i == n_bins - 1 else (x >= lo) & (x < hi)
        if m.sum() > 0:
            cx.append(np.sqrt(lo * hi) if log_x else (lo + hi) / 2)
            cy.append(float(np.median(y[m])))
    ax.plot(cx, cy, "o-", color=MEMENTO_MAGENTA, lw=3.0, markersize=8, zorder=3)
    if log_x:
        ax.set_xscale("log")


def _drop_line(ax, base_vals_per_dr, drs, fmt):
    fracs = [0] + [int(d * 100) for d in drs]
    vals = base_vals_per_dr
    ax.plot(fracs, vals, "o-", color=MEMENTO_MAGENTA, lw=3.0, markersize=9, zorder=3)
    span = (max(vals) - min(vals)) or 1.0
    pad = span * 0.08
    ax.fill_between(fracs, [v - pad for v in vals], [v + pad for v in vals],
                    color=MEMENTO_MAGENTA, alpha=0.07)
    for i, (f, v) in enumerate(zip(fracs, vals)):
        xoff = 6 if i == 0 else 0
        ha = "left" if i == 0 else "center"
        ax.annotate(fmt.format(v), (f, v), textcoords="offset points",
                     xytext=(xoff, 10), ha=ha, fontsize=16, color=MEMENTO_MAGENTA)
    ax.set_xticks(fracs)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.22)


def _two_bar(ax, treat_v, base_v, treat_label, base_label, fmt):
    x = np.arange(2)
    vals = [treat_v, base_v]
    ax.bar(x, vals, 0.55, color=[MEMENTO_MAGENTA, VANILLA_BLUE], alpha=0.9,
           edgecolor="white", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([treat_label, base_label], fontsize=15)
    for xi, v in zip(x, vals):
        ax.annotate(fmt.format(v), (xi, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=16, color=CHARCOAL)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.22)


# ---------------------------------------------------------------------------
# Row drawers
# ---------------------------------------------------------------------------

def draw_corr_row(axes_row, hyp):
    rows = _load_corr(hyp["dir"])["benchmarks"]
    x = np.array([r[hyp["feature"]] for r in rows], dtype=float)
    for col, m in enumerate(METRICS):
        y = np.array([r.get(m["key"], np.nan) for r in rows], dtype=float)
        _scatter_binned(axes_row[col], x, y, log_x=hyp["log_x"])
        axes_row[col].set_xlabel(hyp["xlabel"], fontsize=15)


def _abl_median_at(records, metric_key, base=True, **filt):
    field = ("base_" if base else "treat_") + metric_key
    by_bench = {}
    for r in records:
        if not all(r.get(k) == v for k, v in filt.items()):
            continue
        v = r.get(field)
        if v is None or not np.isfinite(v):
            continue
        by_bench.setdefault(r["bench_id"], []).append(float(v))
    vals = [float(np.median(vs)) for vs in by_bench.values() if vs]
    return float(np.median(vals)) if vals else np.nan


def draw_abl_drop_row(axes_row, hyp):
    j = _load_abl(hyp["dir"])
    recs = j["records"]
    drs = hyp["drs"]
    for col, m in enumerate(METRICS):
        base = _abl_median_at(recs, m["key"], base=True)
        seq = [base] + [_abl_median_at(recs, m["key"], base=False, drop_rate=d) for d in drs]
        _drop_line(axes_row[col], seq, drs, m["fmt"])
        axes_row[col].set_xlabel(hyp["xlabel"], fontsize=15)


def draw_abl_threshold_row(axes_row, hyp):
    j = _load_abl(hyp["dir"])
    recs = [r for r in j["records"] if r.get("threshold") == hyp["thr"]]
    for col, m in enumerate(METRICS):
        base = _abl_median_at(recs, m["key"], base=True)
        treat = _abl_median_at(recs, m["key"], base=False)
        _two_bar(axes_row[col], treat, base, hyp["treat_label"], hyp["base_label"], m["fmt"])


def draw_abl_single_row(axes_row, hyp):
    j = _load_abl(hyp["dir"])
    recs = j["records"]
    for col, m in enumerate(METRICS):
        base = _abl_median_at(recs, m["key"], base=True)
        treat = _abl_median_at(recs, m["key"], base=False)
        _two_bar(axes_row[col], treat, base, hyp["treat_label"], hyp["base_label"], m["fmt"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    apply_double()
    plt.rcParams.update({
        "font.size": 19,
        "axes.titlesize": 23,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })
    fig = plt.figure(figsize=(19.4, 12.8))
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.22],
        left=0.015,
        right=0.99,
        top=0.91,
        bottom=0.075,
        wspace=0.11,
    )
    left_grid = outer[0].subgridspec(
        3, 3, width_ratios=[0.46, 1.0, 1.0], hspace=0.72, wspace=0.36
    )
    right_grid = outer[1].subgridspec(
        4, 3, width_ratios=[0.54, 1.0, 1.0], hspace=0.64, wspace=0.36
    )
    left_label_axes = [fig.add_subplot(left_grid[r, 0]) for r in range(3)]
    right_label_axes = [fig.add_subplot(right_grid[r, 0]) for r in range(4)]
    for ax in left_label_axes + right_label_axes:
        ax.axis("off")
    left_axes = np.array([
        [fig.add_subplot(left_grid[r, c + 1]) for c in range(2)]
        for r in range(3)
    ])
    right_axes = np.array([
        [fig.add_subplot(right_grid[r, c + 1]) for c in range(2)]
        for r in range(4)
    ])

    axes_by_hyp = {}
    label_axes_by_hyp = {}
    for row, hyp in enumerate(CORR_HYPS):
        axes_by_hyp[hyp["id"]] = left_axes[row]
        label_axes_by_hyp[hyp["id"]] = left_label_axes[row]
        draw_corr_row(left_axes[row], hyp)

    for row, hyp in enumerate(ABL_HYPS):
        axes_by_hyp[hyp["id"]] = right_axes[row]
        label_axes_by_hyp[hyp["id"]] = right_label_axes[row]
        if hyp["kind"] == "drop_line":
            draw_abl_drop_row(right_axes[row], hyp)
        elif hyp["kind"] == "threshold_bar":
            draw_abl_threshold_row(right_axes[row], hyp)
        elif hyp["kind"] == "single_bar":
            draw_abl_single_row(right_axes[row], hyp)

    for ax_block in (left_axes, right_axes):
        for col, m in enumerate(METRICS):
            ax_block[0, col].set_title(
                m["col_title"], fontsize=22, fontweight="bold", pad=12,
                color=CHARCOAL,
            )

    row_titles = {
        "H1": "$H_1$ Low-rank\nfit",
        "H2": "$H_2$ Score\nlevel",
        "H3": "$H_3$ Score\nspread",
        "H4": "$H_4$ Target\ncoverage",
        "H5": "$H_5$ Strong-neighbor\npresence",
        "H6": "$H_6$ Neighbor\nsupport",
        "H7": "$H_7$ Same-cat.\nevidence",
    }
    for hyp_id, txt in row_titles.items():
        label_axes_by_hyp[hyp_id].text(
            0.98, 0.5, txt, transform=label_axes_by_hyp[hyp_id].transAxes,
            fontsize=18, fontweight="bold", ha="right", va="center",
            color=CHARCOAL, linespacing=1.05,
        )

    for x, title in [
        (0.265, "Correlational hypotheses"),
        (0.745, "Ablation hypotheses"),
    ]:
        fig.text(
            x, 0.965, title, ha="center", va="center", fontsize=25,
            fontweight="bold", color=CHARCOAL,
        )

    save_fig("bp_predictability_factors_full")


if __name__ == "__main__":
    main()
