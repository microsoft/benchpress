#!/usr/bin/env python
"""Attach calibrated BenchPress intervals and trust probabilities to site data."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONF_DIR = os.path.join(
    REPO_ROOT, "experiments", "sec6_trust", "confidence_calibration")
DATA_PATH = os.path.join(REPO_ROOT, "website", "data.json")
SCORES_PATH = os.path.join(CONF_DIR, "confidence_scores.npz")
RESULTS_PATH = os.path.join(CONF_DIR, "results.json")


def _clean_float(value):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 3)


def _median_by_index(values, index, size, fallback):
    out = np.full(size, fallback, dtype=float)
    for k in range(size):
        selected = values[index == k]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            out[k] = float(np.median(selected))
    return out


def _pava_increasing(values, weights):
    block_values = []
    block_weights = []
    starts = []
    ends = []
    for idx, (value, weight) in enumerate(zip(values, weights, strict=True)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        starts.append(idx)
        ends.append(idx + 1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / merged_weight
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            starts[-2:] = [starts[-2]]
            ends[-2:] = [ends[-1]]

    out = np.empty_like(values, dtype=float)
    for value, start, end in zip(
            block_values, starts, ends, strict=True):
        out[start:end] = value
    return out


def _fit_trust_calibrator(risk, actual, predicted, thresholds, n_bins=20):
    risk = np.asarray(risk, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    finite = (
        np.isfinite(risk)
        & np.isfinite(actual)
        & np.isfinite(predicted)
        & np.isfinite(thresholds)
    )
    risk = risk[finite]
    trusted = (
        np.abs(predicted[finite] - actual[finite]) <= thresholds[finite]
    ).astype(float)
    if risk.size == 0:
        raise ValueError("No finite held-out risk values available.")

    order = np.argsort(risk)
    risk = risk[order]
    trusted = trusted[order]
    bins = np.array_split(np.arange(risk.size), min(n_bins, risk.size))
    centers = np.asarray(
        [float(np.median(risk[indices])) for indices in bins], dtype=float)
    probabilities = np.asarray(
        [float(np.mean(trusted[indices])) for indices in bins], dtype=float)
    weights = np.asarray([float(len(indices)) for indices in bins], dtype=float)
    calibrated = np.clip(
        -_pava_increasing(-probabilities, weights), 0.0, 1.0)

    def predict(risk_values):
        risk_values = np.asarray(risk_values, dtype=float)
        out = np.full(risk_values.shape, np.nan, dtype=float)
        finite_values = np.isfinite(risk_values)
        out[finite_values] = np.interp(
            risk_values[finite_values],
            centers,
            calibrated,
            left=calibrated[0],
            right=calibrated[-1],
        )
        return out

    metadata = {
        "threshold": (
            "10% of the declared benchmark range; "
            "10 raw score units when no finite range is declared"
        ),
        "risk_scale": "threshold-normalized absolute-error risk",
        "num_calibration_cells": int(risk.size),
        "bin_count": int(len(bins)),
        "bin_risk_median": [_clean_float(value) for value in centers],
        "bin_empirical_trust_probability": [
            _clean_float(value) for value in probabilities
        ],
        "bin_calibrated_trust_probability": [
            _clean_float(value) for value in calibrated
        ],
    }
    return predict, metadata


def attach_prediction_intervals(
        data, scores, results, risk_source, results_source):
    """Return site data with calibrated interval and trust fields attached."""
    n_models = len(data["models"])
    n_benchmarks = len(data["benchmarks"])
    matrix_shape = [n_models, n_benchmarks]
    score_metadata = json.loads(str(scores["metadata_json"]))
    if score_metadata.get("matrix_shape") != matrix_shape:
        raise ValueError(
            "Confidence cache matrix mismatch: "
            f"{score_metadata.get('matrix_shape')} != {matrix_shape}")
    if results["setting"].get("matrix_shape") != matrix_shape:
        raise ValueError(
            "Confidence results matrix mismatch: "
            f"{results['setting'].get('matrix_shape')} != {matrix_shape}")

    risk_field = "combined_risk_model_uncertainty"
    risk = np.asarray(scores[risk_field], dtype=float)
    actual = np.asarray(scores["actual"], dtype=float)
    predicted = np.asarray(scores["predicted"], dtype=float)
    test_i = np.asarray(scores["test_i"], dtype=int)
    test_j = np.asarray(scores["test_j"], dtype=int)
    if (
        np.any(test_i < 0) or np.any(test_i >= n_models)
        or np.any(test_j < 0) or np.any(test_j >= n_benchmarks)
    ):
        raise ValueError("Confidence cache contains out-of-range cell indices.")

    method = results["confidence_methods"]["combined_risk_model"]
    conformal_scale = float(method["conformal_90_scale_median"])
    benchmark_bounds = []
    benchmark_thresholds = []
    for benchmark in data["benchmarks"]:
        score_range = (benchmark.get("metric") or {}).get("range")
        if (
            isinstance(score_range, list)
            and len(score_range) == 2
            and score_range[0] is not None
            and score_range[1] is not None
        ):
            lower, upper = map(float, score_range)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(
                    f"Invalid benchmark range for {benchmark['id']}: "
                    f"{score_range}"
                )
            benchmark_bounds.append((lower, upper))
            benchmark_thresholds.append(0.1 * (upper - lower))
        else:
            benchmark_bounds.append(None)
            benchmark_thresholds.append(10.0)
    heldout_thresholds = np.asarray(
        benchmark_thresholds, dtype=float)[test_j]
    normalized_risk = risk / heldout_thresholds
    trust_predictor, trust_metadata = _fit_trust_calibrator(
        normalized_risk,
        actual,
        predicted,
        heldout_thresholds,
    )

    global_normalized_risk = float(np.nanmedian(normalized_risk))
    model_normalized_risk = _median_by_index(
        normalized_risk, test_i, n_models, global_normalized_risk)
    benchmark_normalized_risk = _median_by_index(
        normalized_risk, test_j, n_benchmarks, global_normalized_risk)
    intervals = []
    cell_normalized_risks = []
    for i, row in enumerate(data["predictions"]):
        interval_row = []
        risk_row = []
        for j, prediction in enumerate(row):
            if prediction is None:
                interval_row.append(None)
                risk_row.append(np.nan)
                continue
            point = float(prediction)
            cell_normalized_risk = (
                0.5 * float(model_normalized_risk[i])
                + 0.5 * float(benchmark_normalized_risk[j])
            )
            cell_risk = (
                cell_normalized_risk * float(benchmark_thresholds[j])
            )
            half_width = conformal_scale * cell_risk
            lower = point - half_width
            upper = point + half_width
            if benchmark_bounds[j] is not None:
                bound_lower, bound_upper = benchmark_bounds[j]
                lower = max(bound_lower, lower)
                upper = min(bound_upper, upper)
            interval_row.append([_clean_float(lower), _clean_float(upper)])
            risk_row.append(cell_normalized_risk)
        intervals.append(interval_row)
        cell_normalized_risks.append(risk_row)

    trust_probabilities = trust_predictor(
        np.asarray(cell_normalized_risks, dtype=float))
    benchmark_half_width = (
        conformal_scale
        * benchmark_normalized_risk
        * np.asarray(benchmark_thresholds, dtype=float)
    )
    benchmark_trust = trust_predictor(benchmark_normalized_risk)
    data["prediction_intervals"] = intervals
    data["trust_probabilities"] = [
        [_clean_float(value) for value in row]
        for row in trust_probabilities
    ]
    data.setdefault("meta", {})
    data["meta"]["prediction_interval"] = {
        "method": "Hybrid uncertainty model conformal interval",
        "nominal_coverage": 0.90,
        "heldout_coverage": round(
            float(method["conformal_90_interval"]["coverage"]), 4),
        "heldout_median_width": round(
            float(method["conformal_90_interval"]["median_width"]), 3),
        "conformal_scale_median": round(conformal_scale, 4),
        "risk_source": risk_source,
        "results_source": results_source,
        "risk_field": risk_field,
        "trust_probability": (
            "Calibrated probability that absolute error is within 10% of "
            "the declared benchmark range, or 10 raw units when unavailable"
        ),
        "trust_calibration": trust_metadata,
        "website_estimator": (
            "0.5 * model median threshold-normalized hybrid uncertainty + "
            "0.5 * benchmark median threshold-normalized hybrid uncertainty; "
            "rescaled to the target benchmark"
        ),
        "benchmark_half_width": [
            _clean_float(value) for value in benchmark_half_width
        ],
        "benchmark_trust_probability": [
            _clean_float(value) for value in benchmark_trust
        ],
    }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--scores-path", default=SCORES_PATH)
    parser.add_argument("--results-path", default=RESULTS_PATH)
    args = parser.parse_args()

    with open(args.data_path) as file:
        data = json.load(file)
    with np.load(args.scores_path, allow_pickle=False) as cache:
        scores = {key: cache[key] for key in cache.files}
    with open(args.results_path) as file:
        results = json.load(file)
    attach_prediction_intervals(
        data,
        scores,
        results,
        os.path.relpath(args.scores_path, REPO_ROOT),
        os.path.relpath(args.results_path, REPO_ROOT),
    )
    with open(args.data_path, "w") as file:
        json.dump(data, file, separators=(",", ":"))
        file.write("\n")


if __name__ == "__main__":
    main()
