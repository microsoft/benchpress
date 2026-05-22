#!/usr/bin/env python3
"""Appendix version of the §4.3 benchmark-side composite figure.

Generates two figures with the same 1×5 layout as plot_section51.py
(H3, H4, H5, H6, H7), one per error metric. The file naming
is bp_predictability_factors_51_<metric>.{pdf,png}.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchpress.plot_helpers.visual_identity import (
    CHARCOAL, PEACH, apply_double, save_fig,
)
from benchpress.io_utils import load_json

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_corr(subdir):
    return load_json(os.path.join(HERE, subdir, "results.json"))


def _load_abl(subdir):
    return load_json(os.path.join(HERE, subdir, "ablation_results.json"))


# ---------------------------------------------------------------------------
# Metric-aware helpers
# ---------------------------------------------------------------------------

# config: corr_key (in benchmarks rows), abl_base/abl_treat (in ablation records),
# y-label, value formatter, unit suffix on annotation, "more is" direction word.
METRICS = {
    "medape": {
        "corr_key": "medape",
        "abl_base": "base_medape",
        "abl_treat": "treat_medape",
        "ylabel": "MedAPE (%)",
        "fmt": "{:.1f}",
        "fmt_pad": 0.10,
        "lower_is_better": True,
    },
    "medae": {
        "corr_key": "medae",
        "abl_base": "base_medae",
        "abl_treat": "treat_medae",
        "ylabel": "MedAE (points)",
        "fmt": "{:.2f}",
        "fmt_pad": 0.05,
        "lower_is_better": True,
    },
}


def _scatter_binned(ax, x, y, xlabel, ylabel, title, n_bins=5, log_x=False):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, c=CHARCOAL, s=25, alpha=0.35, zorder=2)

    if log_x:
        x_pos = x[x > 0]
        bins = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), n_bins + 1)
    else:
        bins = np.linspace(x.min(), x.max(), n_bins + 1)
    cx, cy = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (x >= lo) & (x <= hi) if i == n_bins - 1 else (x >= lo) & (x < hi)
        if m.sum() > 0:
            cx.append(np.sqrt(lo * hi) if log_x else (lo + hi) / 2)
            cy.append(float(np.median(y[m])))
    ax.plot(cx, cy, "o-", color=CHARCOAL, zorder=3)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")


def _line_panel(ax, j, drop_rates, mcfg, xlabel, title):
    base = _benchmark_median(j["records"], mcfg["abl_base"])
    vals = [base]
    for d in drop_rates:
        vals.append(_benchmark_median(j["records"], mcfg["abl_treat"], drop_rate=d))

    fracs = [0] + [int(d * 100) for d in drop_rates]
    ax.plot(fracs, vals, "o-", color=CHARCOAL, lw=2.5, markersize=8, zorder=3)
    span = max(vals) - min(vals)
    pad = max(span * 0.10, mcfg["fmt_pad"])
    ax.fill_between(fracs, [v - pad for v in vals], [v + pad for v in vals],
                    color=CHARCOAL, alpha=0.08)
    for i, (f, v) in enumerate(zip(fracs, vals)):
        xoff = 5 if i == 0 else 0
        ha = "left" if i == 0 else "center"
        ax.annotate(mcfg["fmt"].format(v), (f, v), textcoords="offset points",
                    xytext=(xoff, 10), ha=ha, color=CHARCOAL)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(mcfg["ylabel"])
    ax.set_xticks(fracs)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.15)
    ax.set_title(title, fontweight="bold")


def _single_bar_panel(ax, recs, mcfg, base_label, ab_label, title):
    base = _benchmark_median(recs, mcfg["abl_base"])
    treat = _benchmark_median(recs, mcfg["abl_treat"])
    x = np.arange(2)
    w = 0.55
    colors = [PEACH, CHARCOAL]
    labels = [ab_label, base_label]
    vals = [treat, base]

    span = abs(vals[0] - vals[1])
    pad = max(span * 4.0, mcfg["fmt_pad"] * 2)
    lo = min(vals) - pad
    hi = max(vals) + pad
    bars = ax.bar(x, vals, w, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5,
                  bottom=lo)

    for b, v in zip(bars, vals):
        ax.annotate(mcfg["fmt"].format(v),
                    (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", color=CHARCOAL)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(mcfg["ylabel"])
    ax.set_ylim(lo, hi + (hi - lo) * 0.18)
    ax.set_title(title, fontweight="bold")


def _benchmark_median(records, field, **filters):
    by_bench = {}
    for r in records:
        if not all(r.get(k) == v for k, v in filters.items()):
            continue
        v = r.get(field)
        if v is None or not np.isfinite(v):
            continue
        by_bench.setdefault(r["bench_id"], []).append(float(v))
    vals = [float(np.median(vs)) for vs in by_bench.values() if vs]
    return float(np.median(vals)) if vals else np.nan


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_h3(ax, mcfg):
    rows = _load_corr("H3_score_spread")["benchmarks"]
    x = np.array([r["std_score"] for r in rows])
    y = np.array([r[mcfg["corr_key"]] for r in rows])
    _scatter_binned(ax, x, y, "Score std. dev. (log)", mcfg["ylabel"],
                    "$H_3$: Score spread", log_x=True)


def panel_h4(ax, mcfg):
    j = _load_abl("H4_target_coverage")
    _line_panel(ax, j, [0.25, 0.5, 0.75], mcfg,
                "Drop fraction of target obs. (%)",
                "$H_4$: Target coverage")


def panel_h5(ax, mcfg):
    j = _load_abl("H5_strong_neighbor_presence")
    recs = [r for r in j["records"] if r["threshold"] == 0.85]
    _single_bar_panel(ax, recs, mcfg,
                      "W/ strong neighbors", "W/o strong neighbors",
                      "$H_5$: Strong-neighbor presence")


def panel_h6(ax, mcfg):
    j = _load_abl("H6_strong_neighbor_support")
    _line_panel(ax, j, [0.25, 0.5, 0.75], mcfg,
                "Drop fraction of best neighbor's obs. (%)",
                "$H_6$: Strong-neighbor support")


def panel_h7(ax, mcfg):
    j = _load_abl("H7_same_category_evidence")
    recs = j["records"]
    _single_bar_panel(ax, recs, mcfg,
                      "W/ category overlap", "W/o category overlap",
                      "$H_7$: Same-category evidence")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_figure(metric_name):
    mcfg = METRICS[metric_name]
    apply_double()
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 17,
    })
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.0))

    panel_h3(axes[0], mcfg)
    panel_h4(axes[1], mcfg)
    panel_h5(axes[2], mcfg)
    panel_h6(axes[3], mcfg)
    panel_h7(axes[4], mcfg)

    plt.tight_layout(w_pad=0.3, h_pad=1.2)
    save_fig(f"bp_predictability_factors_51_{metric_name}")
    plt.close(fig)


def main():
    for m in METRICS:
        make_figure(m)


if __name__ == "__main__":
    main()
