#!/usr/bin/env python3
"""Composite §4.3 benchmark-side error-analysis figure.

Mirrors the §5.2 style (plot_section52.py): big fonts, conclusion-oriented
titles, scatter + binned trend for correlational panels, grouped-bar (metric
clusters) / line plots for ablation panels.
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
from benchpress.io_utils import load_json

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_corr(subdir):
    return load_json(os.path.join(HERE, subdir, "results.json"))


def _load_abl(subdir):
    return load_json(os.path.join(HERE, subdir, "ablation_results.json"))


def _dual_scatter_binned(ax, x, medape, medae, xlabel, title, log_x=False, quantile=False):
    ax2 = ax.twinx()
    ax.scatter(x, medape, c=MEMENTO_MAGENTA, s=44, alpha=0.34, zorder=2)
    ax2.scatter(x, medae, c=VANILLA_BLUE, s=44, alpha=0.34, zorder=2)
    for axis, vals, color, marker in [
        (ax, medape, MEMENTO_MAGENTA, "o"),
        (ax2, medae, VANILLA_BLUE, "s"),
    ]:
        cx, cy = binned_medians(x, vals, log_x=log_x, quantile=quantile)
        axis.plot(cx, cy, marker + "-", color=color, lw=3.0, markersize=8, zorder=3)
        if log_x:
            axis.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MedAPE (%)")
    ax2.set_ylabel("MedAE (pts)")
    ax.set_title(title, fontweight="bold", pad=9)
    style_dual_axes(ax, ax2)


def _dual_scatter_fit(ax, x, medape, medae, xlabel, title):
    ax2 = ax.twinx()
    ax.scatter(x, medape, c=MEMENTO_MAGENTA, s=44, alpha=0.34, zorder=2)
    ax2.scatter(x, medae, c=VANILLA_BLUE, s=44, alpha=0.34, zorder=2)

    def fit_band(axis, y, color):
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        xf = np.log10(x[mask])
        yf = np.log10(y[mask])
        n = len(xf)
        slope, intercept = np.polyfit(xf, yf, 1)
        yhat = slope * xf + intercept
        resid = yf - yhat
        sigma = np.sqrt(np.sum(resid ** 2) / (n - 2))
        x_center = np.mean(xf)
        sxx = np.sum((xf - x_center) ** 2)

        from scipy.stats import t as tdist
        grid = np.logspace(np.log10(x[mask].min()), np.log10(x[mask].max()), 100)
        gf = np.log10(grid)
        pred = slope * gf + intercept
        tcrit = tdist.ppf(0.975, n - 2)
        se = sigma * np.sqrt(1.0 / n + (gf - x_center) ** 2 / sxx)
        axis.fill_between(grid, 10 ** (pred - tcrit * se), 10 ** (pred + tcrit * se),
                          color=color, alpha=0.12, zorder=2.5, linewidth=0)
        axis.plot(grid, 10 ** pred, "-", color=color, lw=3.0, zorder=3)

    fit_band(ax, medape, MEMENTO_MAGENTA)
    fit_band(ax2, medae, VANILLA_BLUE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MedAPE (log %)")
    ax2.set_ylabel("MedAE (log pts)")
    ax.set_title(title, fontweight="bold", pad=9)
    style_dual_axes(ax, ax2)


def _dual_line(ax, x, medape, medae, xlabel, title, xticklabels=None):
    ax2 = ax.twinx()
    ax.plot(x, medape, "o-", color=MEMENTO_MAGENTA, lw=3.0, markersize=8, zorder=3)
    ax2.plot(x, medae, "s--", color=VANILLA_BLUE, lw=3.0, markersize=8, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MedAPE (%)")
    ax2.set_ylabel("MedAE (pts)")
    ax.set_xticks(x)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.set_title(title, fontweight="bold", pad=9)
    style_dual_axes(ax, ax2)


def _scatter_fit(ax, x, y, xlabel, title, log_x=False, ci=0.95):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, c=CYAN_TEAL, s=48, alpha=0.45, zorder=2)

    xf = np.log10(x) if log_x else x.astype(float)
    n = len(xf)
    slope, intercept = np.polyfit(xf, y, 1)
    yhat = slope * xf + intercept
    resid = y - yhat
    sigma = np.sqrt(np.sum(resid ** 2) / (n - 2))
    x_center = np.median(xf)
    Sxx = np.sum((xf - x_center) ** 2)

    if log_x:
        grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        gf = np.log10(grid)
    else:
        grid = np.linspace(x.min(), x.max(), 100)
        gf = grid
    pred = slope * gf + intercept
    from scipy.stats import t as tdist
    tcrit = tdist.ppf(0.5 + ci / 2, n - 2)
    se = sigma * np.sqrt(1.0 / n + (gf - x_center) ** 2 / Sxx)
    lo, hi = pred - tcrit * se, pred + tcrit * se

    ax.fill_between(grid, lo, hi, color=MEMENTO_MAGENTA, alpha=0.12, zorder=2.5)
    ax.plot(grid, pred, color=MEMENTO_MAGENTA, lw=3.2, zorder=3)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MedAPE (%)")
    ax.set_title(title, fontweight="bold", pad=9)


def panel_h3(ax):
    rows = _load_corr("H3_score_spread")["benchmarks"]
    x = np.array([r["std_score"] for r in rows])
    medape = np.array([r["medape"] for r in rows])
    medae = np.array([r["medae"] for r in rows])
    _dual_scatter_fit(
        ax, x, medape, medae, "Score std. dev. (log)",
        "$H_3$: Wider score ranges\nincrease error",
    )


def _grouped_bar(ax, base_medape, base_medae, ab_medape, ab_medae,
                 base_label, ab_label, title):
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [base_medape, base_medae], w,
           color=VANILLA_BLUE, alpha=0.9, label=base_label,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, [ab_medape, ab_medae], w,
           color=MEMENTO_MAGENTA, alpha=0.9, label=ab_label,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["MedAPE", "MedAE"])
    ymax = max(base_medape, base_medae, ab_medape, ab_medae)
    ax.set_ylim(0, ymax * 1.58)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98),
              ncol=2, handletextpad=0.25, columnspacing=0.55,
              borderaxespad=0.0, fontsize=20)
    ax.set_title(title, fontweight="bold", pad=9)


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


def panel_h4(ax):
    """H4 target coverage: line across drop fractions."""
    j = _load_abl("H4_target_coverage")
    drs = [0.25, 0.5, 0.75]
    base_medape = _benchmark_median(j["records"], "base_medape")
    base_medae = _benchmark_median(j["records"], "base_medae")
    medape = [base_medape]
    medae = [base_medae]
    for d in drs:
        medape.append(_benchmark_median(j["records"], "treat_medape", drop_rate=d))
        medae.append(_benchmark_median(j["records"], "treat_medae", drop_rate=d))

    fracs = [0] + [int(d * 100) for d in drs]
    _dual_line(
        ax, fracs, medape, medae, "Target drop (%)",
        "$H_4$: More target\nobservations reduce error",
    )


def panel_h5(ax):
    """H5 strong-neighbor presence: grouped bar at headline |r|=0.85."""
    j = _load_abl("H5_strong_neighbor_presence")
    recs = [r for r in j["records"] if r["threshold"] == 0.85]
    base_medape = _benchmark_median(recs, "base_medape")
    base_medae = _benchmark_median(recs, "base_medae")
    treat_medape = _benchmark_median(recs, "treat_medape")
    treat_medae = _benchmark_median(recs, "treat_medae")
    _grouped_bar(ax, base_medape, base_medae, treat_medape, treat_medae,
                 "Present", "Removed",
                 "$H_5$: Strong-neighbor presence\nreduces error")


def panel_h6(ax):
    """H6 neighbor support: dual-axis line across drop fractions of best peer."""
    j = _load_abl("H6_strong_neighbor_support")
    drs = [0.25, 0.5, 0.75]
    base_medape = _benchmark_median(j["records"], "base_medape")
    base_medae = _benchmark_median(j["records"], "base_medae")
    medape = [base_medape]
    medae = [base_medae]
    for d in drs:
        medape.append(_benchmark_median(j["records"], "treat_medape", drop_rate=d))
        medae.append(_benchmark_median(j["records"], "treat_medae", drop_rate=d))

    fracs = [0] + [int(d * 100) for d in drs]
    _dual_line(
        ax, fracs, medape, medae, "Neighbor drop (%)",
        "$H_6$: More neighbor\nevidence reduces error",
    )


def panel_h7(ax):
    """H7 same-category evidence: grouped bar Full vs no same-category."""
    j = _load_abl("H7_same_category_evidence")
    recs = j["records"]
    base_medape = _benchmark_median(recs, "base_medape")
    base_medae = _benchmark_median(recs, "base_medae")
    treat_medape = _benchmark_median(recs, "treat_medape")
    treat_medae = _benchmark_median(recs, "treat_medae")
    _grouped_bar(ax, base_medape, base_medae, treat_medape, treat_medae,
                 "W/ category overlap", "W/o category overlap",
                 "$H_7$: Category overlap\nLittle effect")


def main():
    apply_double()
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 23,
        "axes.labelsize": 23,
        "xtick.labelsize": 20,
        "ytick.labelsize": 18,
        "legend.fontsize": 20,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
    axes = axes.ravel()

    # Main-text figure shows the jointly supported benchmark-side patterns.
    # H1, H2, H6, and H7 remain in the table and appendix grid.
    panel_h3(axes[0])
    panel_h4(axes[1])
    panel_h5(axes[2])

    fig.subplots_adjust(left=0.070, right=0.965, bottom=0.20, top=0.79,
                        wspace=0.56)
    save_fig("bp_predictability_factors_51")


if __name__ == "__main__":
    main()
