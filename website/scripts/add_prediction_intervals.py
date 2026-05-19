#!/usr/bin/env python
"""Attach calibrated BenchPress prediction intervals to website/data.json.

The confidence experiment only evaluates held-out observed cells. For the public
website, we reuse its cross-fit combined-risk scores as calibration data:
per-cell intervals combine benchmark-level and model-level median risk, then
apply the leave-fold-out conformal scale reported by Section 4.4. Trust
probabilities are calibrated from the same held-out cells for the event
``abs(predicted - actual) <= 10``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SITE_DIR = Path(__file__).resolve().parents[1]
BENCHPRESS_DIR = SITE_DIR.parent / "github"
CONF_DIR = BENCHPRESS_DIR / "experiments/sec4_building_benchpress/confidence_calibration"
DATA_PATH = SITE_DIR / "data.json"
SCORES_PATH = CONF_DIR / "confidence_scores.npz"
RESULTS_PATH = CONF_DIR / "results.json"


def _clean_float(x: float | None) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), 3)


def _median_by_index(values: np.ndarray, index: np.ndarray, size: int, fallback: float) -> np.ndarray:
    out = np.full(size, fallback, dtype=float)
    for k in range(size):
        vals = values[index == k]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[k] = float(np.median(vals))
    return out


def _pava_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators algorithm for nondecreasing values."""
    values: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for idx, (val, weight) in enumerate(zip(y, w, strict=True)):
        values.append(float(val))
        weights.append(float(weight))
        starts.append(idx)
        ends.append(idx + 1)
        while len(values) >= 2 and values[-2] > values[-1]:
            merged_weight = weights[-2] + weights[-1]
            merged_value = (values[-2] * weights[-2] + values[-1] * weights[-1]) / merged_weight
            values[-2:] = [merged_value]
            weights[-2:] = [merged_weight]
            starts[-2:] = [starts[-2]]
            ends[-2:] = [ends[-1]]

    out = np.empty_like(y, dtype=float)
    for val, start, end in zip(values, starts, ends, strict=True):
        out[start:end] = val
    return out


def _fit_trust_calibrator(
    risk: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold: float = 10.0,
    n_bins: int = 20,
):
    """Calibrate P(abs error <= threshold | risk) from held-out cells."""
    risk = np.asarray(risk, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(risk) & np.isfinite(actual) & np.isfinite(predicted)
    risk = risk[finite]
    trusted = (np.abs(predicted[finite] - actual[finite]) <= threshold).astype(float)
    if risk.size == 0:
        raise ValueError("No finite held-out risk values available for trust calibration.")

    order = np.argsort(risk)
    risk = risk[order]
    trusted = trusted[order]
    bins = np.array_split(np.arange(risk.size), min(n_bins, risk.size))
    centers = np.array([float(np.median(risk[b])) for b in bins], dtype=float)
    probs = np.array([float(np.mean(trusted[b])) for b in bins], dtype=float)
    weights = np.array([float(len(b)) for b in bins], dtype=float)

    calibrated_probs = -_pava_increasing(-probs, weights)
    calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)

    def predict(risk_values: np.ndarray) -> np.ndarray:
        risk_values = np.asarray(risk_values, dtype=float)
        out = np.full(risk_values.shape, np.nan, dtype=float)
        finite_values = np.isfinite(risk_values)
        out[finite_values] = np.interp(
            risk_values[finite_values],
            centers,
            calibrated_probs,
            left=calibrated_probs[0],
            right=calibrated_probs[-1],
        )
        return out

    metadata = {
        "threshold": threshold,
        "num_calibration_cells": int(risk.size),
        "bin_count": int(len(bins)),
        "bin_risk_median": [_clean_float(x) for x in centers],
        "bin_empirical_trust_probability": [_clean_float(x) for x in probs],
        "bin_calibrated_trust_probability": [_clean_float(x) for x in calibrated_probs],
    }
    return predict, metadata


def _percent_like_columns(observed: list[list[float | None]]) -> list[bool]:
    arr = np.array([[np.nan if v is None else float(v) for v in row] for row in observed], dtype=float)
    cols = []
    for j in range(arr.shape[1]):
        vals = arr[:, j]
        vals = vals[np.isfinite(vals)]
        cols.append(bool(vals.size and vals.min() >= -1.0 and vals.max() <= 101.0))
    return cols


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    results = json.loads(RESULTS_PATH.read_text())
    scores = np.load(SCORES_PATH, allow_pickle=False)

    risk_field = "combined_risk_model_uncertainty"
    risk = scores[risk_field].astype(float)
    actual = scores["actual"].astype(float)
    predicted = scores["predicted"].astype(float)
    test_i = scores["test_i"].astype(int)
    test_j = scores["test_j"].astype(int)

    n_models = len(data["models"])
    n_bench = len(data["benchmarks"])
    global_risk = float(np.nanmedian(risk))
    model_risk = _median_by_index(risk, test_i, n_models, global_risk)
    benchmark_risk = _median_by_index(risk, test_j, n_bench, global_risk)

    method = results["confidence_methods"]["combined_risk_model"]
    conformal_scale = float(method["conformal_90_scale_median"])
    percent_like = _percent_like_columns(data["observed"])
    trust_predictor, trust_metadata = _fit_trust_calibrator(risk, actual, predicted)

    intervals = []
    cell_risks = []
    for i, row in enumerate(data["predictions"]):
        out_row = []
        risk_row = []
        for j, pred in enumerate(row):
            if pred is None:
                out_row.append(None)
                risk_row.append(np.nan)
                continue
            point = float(pred)
            cell_risk = 0.5 * float(model_risk[i]) + 0.5 * float(benchmark_risk[j])
            risk_row.append(cell_risk)
            half_width = conformal_scale * cell_risk
            lower = point - half_width
            upper = point + half_width
            if percent_like[j]:
                lower = max(0.0, lower)
                upper = min(100.0, upper)
            out_row.append([_clean_float(lower), _clean_float(upper)])
        intervals.append(out_row)
        cell_risks.append(risk_row)

    benchmark_half_width = (conformal_scale * benchmark_risk).tolist()
    trust_probabilities = trust_predictor(np.array(cell_risks, dtype=float))
    benchmark_trust_probabilities = trust_predictor(benchmark_risk)
    data["prediction_intervals"] = intervals
    data.pop("confidence_scores", None)
    data["trust_probabilities"] = [
        [_clean_float(x) for x in row]
        for row in trust_probabilities
    ]
    data.setdefault("meta", {})
    data["meta"]["prediction_interval"] = {
        "method": "Hybrid uncertainty model conformal interval",
        "nominal_coverage": 0.90,
        "heldout_coverage": round(float(method["conformal_90_interval"]["coverage"]), 4),
        "heldout_median_width": round(float(method["conformal_90_interval"]["median_width"]), 3),
        "conformal_scale_median": round(conformal_scale, 4),
        "risk_source": "github/experiments/sec4_building_benchpress/confidence_calibration/confidence_scores.npz",
        "risk_field": risk_field,
        "trust_probability": "Calibrated P(abs(predicted - actual) <= 10 score points | hybrid uncertainty risk)",
        "trust_calibration": trust_metadata,
        "website_estimator": "0.5 * model median hybrid uncertainty + 0.5 * benchmark median hybrid uncertainty",
        "benchmark_half_width": [_clean_float(x) for x in benchmark_half_width],
        "benchmark_trust_probability": [_clean_float(x) for x in benchmark_trust_probabilities],
    }

    DATA_PATH.write_text(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    main()
