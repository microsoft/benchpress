#!/usr/bin/env python3
"""Statistical inference helpers for BenchPress experiments.

Single home for paired Wilcoxon, bootstrap CIs, and any future
significance-testing utilities. Metric definitions live in
`benchpress.evaluation_harness`; this module only consumes
already-computed per-record metric deltas and aggregates / tests them.
"""
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats


def median_metric(entries, key="medape"):
    """Median finite metric value from a list of result dictionaries."""
    vals = [
        entry[key]
        for entry in entries
        if entry.get(key) is not None and np.isfinite(entry[key])
    ]
    return float(np.median(vals)) if vals else np.nan


def wilcoxon_signed_rank(
    values,
    *,
    min_n=5,
    alternative="two-sided",
    drop_zeros_for_test=False,
    invalid_median=float("nan"),
    invalid_p=float("nan"),
    p_key="p_value",
    include_sign_counts=False,
):
    """Two-sided Wilcoxon signed-rank summary against zero.

    `values` are already-paired deltas (for example, ablation minus baseline,
    or one per-benchmark/per-model median delta). Non-finite values are removed.
    Some older scripts dropped exact zeros before the test while still reporting
    `n` and median over all finite deltas; `drop_zeros_for_test=True` preserves
    that convention.
    """
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    test_arr = arr[arr != 0] if drop_zeros_for_test else arr

    if len(test_arr) >= min_n and np.any(test_arr != 0):
        _, p = sp_stats.wilcoxon(test_arr, alternative=alternative)
        out = {
            "median_delta": float(np.median(arr)),
            p_key: float(p),
            "n": int(len(arr)),
        }
    else:
        out = {
            "median_delta": float(invalid_median),
            p_key: float(invalid_p),
            "n": int(len(arr)),
        }
    if include_sign_counts:
        out["n_positive"] = int((arr > 0).sum())
        out["n_negative"] = int((arr < 0).sum())
    return out


def paired_wilcoxon(vals_a, vals_b):
    """Paired Wilcoxon signed-rank test. Returns (median_diff, p_value)."""
    diffs = np.array(vals_a, dtype=float) - np.array(vals_b, dtype=float)
    median_diff = float(np.median(diffs))
    if np.all(diffs == 0):
        return median_diff, 1.0
    summary = wilcoxon_signed_rank(
        diffs,
        min_n=1,
        p_key="p",
        invalid_median=median_diff,
        invalid_p=float("nan"),
    )
    return summary["median_delta"], summary["p"]


def paired_wilcoxon_by_metric(base, abl, metric):
    """Paired Wilcoxon on Δmetric = ablation − baseline for matched ids."""
    deltas = [
        abl[i][metric] - base[i][metric]
        for i in sorted(set(base) & set(abl))
    ]
    return wilcoxon_signed_rank(
        deltas,
        p_key="p",
        include_sign_counts=True,
    )


def wilcoxon_grouped_median(
    records,
    metrics,
    *,
    group_key,
    delta_key_template="delta_{metric}",
    min_groups=5,
    invalid_median=float("nan"),
    invalid_p=float("nan"),
    p_key="p_value",
    include_sign_counts=False,
    drop_zeros_for_test=False,
):
    """Wilcoxon vs 0 after taking one median Δ per group.

    For each metric, group `records` by `group_key`, take the median of
    `delta_<metric>` across seeds within each group, then
    run a two-sided Wilcoxon signed-rank test against 0 on the resulting
    per-group medians. Returns a dict keyed by metric:

        {metric: {"median_delta": float, "p_value": float, "n": int}}

    NaN-valued and non-finite deltas are dropped before taking the per-group
    median.
    """
    by_group = defaultdict(list)
    for r in records:
        by_group[r[group_key]].append(r)

    out = {}
    for metric in metrics:
        delta_key = delta_key_template.format(metric=metric)
        per_group_medians = []
        for recs in by_group.values():
            vals = [r.get(delta_key) for r in recs]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                per_group_medians.append(float(np.median(vals)))
        out[metric] = wilcoxon_signed_rank(
            per_group_medians,
            min_n=min_groups,
            invalid_median=invalid_median,
            invalid_p=invalid_p,
            p_key=p_key,
            include_sign_counts=include_sign_counts,
            drop_zeros_for_test=drop_zeros_for_test,
        )
    return out


def wilcoxon_per_benchmark(records, metrics, **kwargs):
    """Wilcoxon vs 0 on one median Δ per benchmark."""
    return wilcoxon_grouped_median(records, metrics, group_key="bench_id", **kwargs)
