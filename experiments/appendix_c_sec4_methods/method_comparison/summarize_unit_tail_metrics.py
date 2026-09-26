#!/usr/bin/env python3
"""Summarize percentage-only and tail errors for ICLR headline results."""

import argparse
import gzip
import hashlib
import json
import os

import numpy as np

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_METRICS,
    M_FULL,
    compute_prediction_error,
    compute_prediction_tail_metrics,
    percentage_benchmark_mask,
)
from benchpress.io_utils import write_json_atomic

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_METHOD_DIR = os.path.join(
    REPO_ROOT, "experiments", "sec4_building_benchpress", "method_comparison")
DEFAULT_PROBE_DIR = os.path.join(
    REPO_ROOT, "experiments", "sec5_findings", "optimal_probe", "all_known", "results")
DEFAULT_PROSPECTIVE_RAW = os.path.join(
    REPO_ROOT, "experiments", "sec5_findings", "prospective_update", "results",
    "may_to_aug", "raw_predictions.json.gz")
DEFAULT_OUT = os.path.join(HERE, "unit_tail_metrics_summary.json")


def _load_json_auto(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _round_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, int):
            out[key] = int(value)
        elif isinstance(value, float):
            out[key] = _finite_or_none(value)
        else:
            out[key] = value
    return out


def _group_median_metrics(actual, predicted, groups, abs_error_threshold):
    groups = np.asarray(groups)
    rows = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if not np.any(mask):
            continue
        metrics = compute_prediction_tail_metrics(
            actual[mask], predicted[mask], abs_error_threshold=abs_error_threshold)
        if metrics["n"] > 0:
            rows.append(metrics)
    if not rows:
        empty = compute_prediction_tail_metrics([], [], abs_error_threshold=abs_error_threshold)
        return empty
    out = {"n": int(sum(row["n"] for row in rows))}
    for key in ("medae", "medape", "p90_abs_error", "frac_abs_error_gt_threshold"):
        values = [row[key] for row in rows if np.isfinite(row[key])]
        out[key] = float(np.median(values)) if values else float("nan")
    out["abs_error_threshold"] = float(abs_error_threshold)
    return out


def _summarize_arrays(actual, predicted, benchmark_indices, *, groups=None,
                      abs_error_threshold=10.0):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    benchmark_indices = np.asarray(benchmark_indices, dtype=int)
    if not (len(actual) == len(predicted) == len(benchmark_indices)):
        raise ValueError("actual, predicted, and benchmark_indices must align")
    pct_mask = percentage_benchmark_mask(
        benchmark_indices, matrix=M_FULL, metric=BENCH_METRICS,
        benchmark_ids=BENCH_IDS)
    pooled_all = compute_prediction_tail_metrics(
        actual, predicted, abs_error_threshold=abs_error_threshold)
    pooled_pct = compute_prediction_tail_metrics(
        actual[pct_mask], predicted[pct_mask], abs_error_threshold=abs_error_threshold)
    if groups is None:
        paper_all = pooled_all
        paper_pct = pooled_pct
        aggregation = "pooled"
    else:
        groups = np.asarray(groups)
        paper_all = _group_median_metrics(
            actual, predicted, groups, abs_error_threshold)
        paper_pct = _group_median_metrics(
            actual[pct_mask], predicted[pct_mask], groups[pct_mask], abs_error_threshold)
        aggregation = "median over groups"
    table = {
        "medae_all": paper_all["medae"],
        "medae_pct_only": paper_pct["medae"],
        "medape_all": paper_all["medape"],
        "medape_pct_only": paper_pct["medape"],
        "p90_abs_err_all": pooled_all["p90_abs_error"],
        "p90_abs_err_pct_only": pooled_pct["p90_abs_error"],
        "frac_abs_err_gt_10_all": pooled_all["frac_abs_error_gt_threshold"],
        "frac_abs_err_gt_10_pct_only": pooled_pct["frac_abs_error_gt_threshold"],
        "n_cells_all": pooled_all["n"],
        "n_cells_pct_only": pooled_pct["n"],
    }
    return {
        "aggregation_for_medae_medape": aggregation,
        "table": _round_metrics(table),
        "pooled_all": _round_metrics(pooled_all),
        "pooled_pct_only": _round_metrics(pooled_pct),
        "paper_aggregation_all": _round_metrics(paper_all),
        "paper_aggregation_pct_only": _round_metrics(paper_pct),
    }


def _cell_key_hash(keys):
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(json.dumps(key, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _method_prediction_path(method_dir, transform, method):
    results = _load_json_auto(os.path.join(method_dir, "results.json"))
    rel = results[transform][method]["prediction_file"]
    path = os.path.join(method_dir, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing prediction file for {transform} {method}: {path}")
    return path


def _load_method_arrays(method_dir, transform, method):
    path = _method_prediction_path(method_dir, transform, method)
    with np.load(path, allow_pickle=False) as data:
        return {
            "path": path,
            "actual": np.asarray(data["actual"], dtype=float),
            "predicted": np.asarray(data["predicted"], dtype=float),
            "benchmark_indices": np.asarray(data["test_j"], dtype=int),
            "fold_id": np.asarray(data["fold_id"], dtype=int),
        }


def summarize_method_comparison(method_dir, abs_error_threshold):
    results = _load_json_auto(os.path.join(method_dir, "results.json"))
    configs = [
        ("method_comparison", "\\benchpress{}", "logit", "Bias ALS"),
        ("method_comparison", "logit-space model mean", "logit", "Model Mean"),
    ]
    entries = []
    reference_keys = None
    for setting, label, transform, method in configs:
        arrays = _load_method_arrays(method_dir, transform, method)
        keys = [
            (int(fold), int(j), int(row_index))
            for row_index, (fold, j) in enumerate(zip(arrays["fold_id"], arrays["benchmark_indices"]))
        ]
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("method-comparison predictors are not aligned on row order")
        summary = _summarize_arrays(
            arrays["actual"], arrays["predicted"], arrays["benchmark_indices"],
            groups=arrays["fold_id"], abs_error_threshold=abs_error_threshold)
        headline = results[transform][method]
        summary["headline_paper_all"] = {
            "medae": float(headline["medae_median"]),
            "medape": float(headline["medape_median"]),
        }
        summary["table"]["medae_all"] = float(headline["medae_median"])
        summary["table"]["medape_all"] = float(headline["medape_median"])
        entries.append({
            "setting": setting,
            "method": label,
            "source_path": os.path.relpath(arrays["path"], REPO_ROOT),
            "same_cell_key_sha256": _cell_key_hash(keys),
            **summary,
        })
    return entries


def _rows_to_arrays(rows):
    actual = []
    predicted = []
    benchmark_indices = []
    keys = []
    for row in rows:
        actual.append(row.get("actual"))
        predicted.append(row.get("pred"))
        if row.get("benchmark_index") is not None:
            j = int(row["benchmark_index"])
        else:
            j = BENCH_IDS.index(row["benchmark_id"]) if row.get("benchmark_id") in BENCH_IDS else -1
        benchmark_indices.append(j)
        keys.append((
            row.get("target_model_index", row.get("model_id")),
            row.get("benchmark_index", row.get("benchmark_id")),
            row.get("k"),
            row.get("ordering"),
        ))
    return np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float), np.asarray(benchmark_indices, dtype=int), keys


def _probe_hidden_rows(path, k):
    payload = _load_json_auto(path)
    rows = [
        row for row in payload["raw_predictions"]
        if int(row["k"]) == int(k) and not row.get("is_revealed", False)
    ]
    return rows


def summarize_probe_selection(probe_dir, abs_error_threshold):
    configs = [
        ("probe_selection_in_sample_cost_unaware", "\\benchpress{}", "fixed_order_medae_any_hidden_only.json.gz"),
        ("probe_selection_in_sample_cost_unaware", "logit-space model mean", "fixed_order_medae_any_logit_model_mean_hidden_only.json.gz"),
        ("probe_selection_in_sample_cost_aware", "\\benchpress{}", "fixed_order_medae_low_cost_hidden_only.json.gz"),
        ("probe_selection_in_sample_cost_aware", "logit-space model mean", "fixed_order_medae_low_cost_logit_model_mean_hidden_only.json.gz"),
    ]
    entries = []
    reference_by_setting = {}
    for setting, label, filename in configs:
        path = os.path.join(probe_dir, filename)
        rows = _probe_hidden_rows(path, k=5)
        actual, predicted, benchmark_indices, keys = _rows_to_arrays(rows)
        key_hash = _cell_key_hash(keys)
        if setting not in reference_by_setting:
            reference_by_setting[setting] = key_hash
        elif key_hash != reference_by_setting[setting]:
            raise ValueError(f"probe-selection rows do not align for {setting}")
        summary = _summarize_arrays(
            actual, predicted, benchmark_indices,
            abs_error_threshold=abs_error_threshold)
        entries.append({
            "setting": setting,
            "method": label,
            "source_path": os.path.relpath(path, REPO_ROOT),
            "same_cell_key_sha256": key_hash,
            **summary,
        })
    return entries


def _prospective_rows(path, method, ordering, k):
    rows = _load_json_auto(path)
    return [
        row for row in rows
        if row.get("part") == "new_models_frozen_probe_set"
        and row.get("method") == method
        and row.get("ordering") == ordering
        and int(row.get("k")) == int(k)
        and not row.get("is_probe", False)
    ]


def summarize_prospective(raw_path, abs_error_threshold):
    configs = [
        ("prospective_new_models", "\\benchpress{}", "benchpress_fixed_order"),
        ("prospective_new_models", "logit-space model mean", "logit_model_mean_fixed_order"),
    ]
    entries = []
    reference_hash = None
    for setting, label, method in configs:
        rows = _prospective_rows(raw_path, method=method, ordering="medae_any", k=5)
        actual, predicted, benchmark_indices, keys = _rows_to_arrays(rows)
        key_hash = _cell_key_hash(keys)
        if reference_hash is None:
            reference_hash = key_hash
        elif key_hash != reference_hash:
            raise ValueError("prospective rows do not align")
        summary = _summarize_arrays(
            actual, predicted, benchmark_indices,
            abs_error_threshold=abs_error_threshold)
        entries.append({
            "setting": setting,
            "method": label,
            "source_path": os.path.relpath(raw_path, REPO_ROOT),
            "same_cell_key_sha256": key_hash,
            **summary,
        })
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-dir", default=DEFAULT_METHOD_DIR)
    parser.add_argument("--probe-dir", default=DEFAULT_PROBE_DIR)
    parser.add_argument("--prospective-raw", default=DEFAULT_PROSPECTIVE_RAW)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--abs-error-threshold", type=float, default=10.0)
    args = parser.parse_args()

    entries = []
    entries.extend(summarize_method_comparison(args.method_dir, args.abs_error_threshold))
    entries.extend(summarize_probe_selection(args.probe_dir, args.abs_error_threshold))
    entries.extend(summarize_prospective(args.prospective_raw, args.abs_error_threshold))
    output = {
        "metric_definitions": {
            "medae": "median absolute error in original score units; method comparison uses the paper's median over the 30 held-out folds",
            "medape": "median absolute percentage error; true scores with |score| <= 1e-6 are omitted from APE before taking medians",
            "pct_only": "cells whose benchmark metadata declares a percentage scale or [0,100] range via benchpress.methods.transforms._is_pct_bench",
            "p90_abs_error": "90th percentile of absolute error pooled over the same prediction rows",
            "frac_abs_err_gt_10": "fraction of finite predictions with absolute error greater than 10 score points",
        },
        "matrix": {
            "shape": [int(M_FULL.shape[0]), int(M_FULL.shape[1])],
            "n_observed": int(np.isfinite(M_FULL).sum()),
            "n_benchmarks": int(len(BENCH_IDS)),
        },
        "entries": entries,
    }
    write_json_atomic(args.out, output, indent=2)
    print(f"Saved -> {args.out}")
    for entry in entries:
        table = entry["table"]
        print(
            f"{entry['setting']} | {entry['method']} | "
            f"MedAE {table['medae_all']:.3f} / pct {table['medae_pct_only']:.3f} | "
            f"P90 {table['p90_abs_err_all']:.3f} / pct {table['p90_abs_err_pct_only']:.3f} | "
            f"n {table['n_cells_all']}/{table['n_cells_pct_only']}"
        )


if __name__ == "__main__":
    main()
