#!/usr/bin/env python
"""Build the deterministic BenchPress website snapshot and identity report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from benchpress.build_benchmark_matrix.build_benchmark_matrix import (
    _LEGACY_SCORES,
)
from benchpress.evaluation_harness import (
    BENCH_METRICS,
    BENCH_IDS,
    M_FULL,
    MODEL_IDS,
    benchmark_metric_identity_sha256,
    matrix_identity_sha256,
)
from benchpress.methods.predictors import predict_benchpress_scores
from website.scripts.add_prediction_intervals import (
    attach_prediction_intervals,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MATRIX_PATH = os.path.join(
    REPO_ROOT, "benchpress", "data", "llm_benchmark_data.json")
METHOD_DIR = os.path.join(
    REPO_ROOT, "experiments", "sec4_building_benchpress", "method_comparison")
METHOD_RESULTS_PATH = os.path.join(METHOD_DIR, "results.json")
CONF_DIR = os.path.join(
    REPO_ROOT, "experiments", "sec6_trust", "confidence_calibration")
CONFIDENCE_SCORES_PATH = os.path.join(CONF_DIR, "confidence_scores.npz")
CONFIDENCE_RESULTS_PATH = os.path.join(CONF_DIR, "results.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "website", "data.json")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observed_identity_hash(matrix):
    digest = hashlib.sha256()
    for i, model_id in enumerate(MODEL_IDS):
        for j, benchmark_id in enumerate(BENCH_IDS):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            digest.update(
                f"{model_id}\t{benchmark_id}\t{float(value):.17g}\n".encode())
    return digest.hexdigest()


def _matrix_to_json(matrix, digits=None):
    output = []
    for row in matrix:
        output.append([
            None if not np.isfinite(value)
            else (
                float(value) if digits is None
                else round(float(value), digits)
            )
            for value in row
        ])
    return output


def _source_payload(score):
    return {
        "url": score.get("reference_url") or score.get("source_url"),
        "type": score.get("source_type"),
        "matches_canonical": score.get("matches_canonical"),
        "reported_setting": score.get("reported_setting"),
        "audit_status": score.get("audit_status"),
        "notes": score.get("notes"),
    }


def _benchmark_metric(benchmark, values):
    setting = benchmark.get("canonical_setting") or {}
    finite = values[np.isfinite(values)]
    percent_like = bool(
        finite.size and finite.min() >= -1.0 and finite.max() <= 101.0)
    metric_type = setting.get("metric_type") or (
        "pct" if percent_like else "score")
    value_range = setting.get("range")
    if value_range is None and metric_type == "pct":
        value_range = [0, 100]
    if value_range is not None and finite.size:
        if finite.min() < value_range[0] or finite.max() > value_range[1]:
            raise ValueError(
                f"{benchmark['id']} contains scores outside canonical range "
                f"{value_range}: [{finite.min()}, {finite.max()}]"
            )
    return {
        "type": metric_type,
        "range": value_range,
        "higher_is_better": setting.get("higher_is_better", True),
    }


def _raw_prediction_validation(path, label):
    with np.load(path, allow_pickle=False) as data:
        fold_id = data["fold_id"].astype(int)
        test_i = data["test_i"].astype(int)
        test_j = data["test_j"].astype(int)
        actual = data["actual"].astype(float)
        predicted = data["predicted"].astype(float)
        metadata = json.loads(str(data["metadata_json"]))

    expected = M_FULL[test_i, test_j]
    differences = np.abs(actual - expected)
    mismatch = ~np.isclose(actual, expected, rtol=0.0, atol=0.0)
    flat = test_i * len(BENCH_IDS) + test_j
    unique, counts = np.unique(flat, return_counts=True)
    observed_flat = np.flatnonzero(np.isfinite(M_FULL).ravel())
    missing_cells = np.setdiff1d(observed_flat, unique)
    extra_cells = np.setdiff1d(unique, observed_flat)
    repetitions = sorted(set(int(value) for value in counts))
    validation = {
        "label": label,
        "path": os.path.relpath(path, REPO_ROOT),
        "matrix_shape": metadata.get("matrix_shape"),
        "raw_predictions": int(len(actual)),
        "unique_observed_cells": int(len(unique)),
        "repetitions_per_cell": repetitions,
        "missing_observed_cells": int(len(missing_cells)),
        "extra_cells": int(len(extra_cells)),
        "mismatch_count": int(np.sum(mismatch)),
        "max_absolute_difference": (
            0.0 if differences.size == 0 else float(np.max(differences))
        ),
    }
    if (
        metadata.get("matrix_shape") != list(M_FULL.shape)
        or validation["unique_observed_cells"] != int(np.isfinite(M_FULL).sum())
        or validation["repetitions_per_cell"] != [10]
        or validation["missing_observed_cells"] != 0
        or validation["extra_cells"] != 0
        or validation["mismatch_count"] != 0
    ):
        raise ValueError(f"{label} raw-score identity validation failed.")
    return validation, {
        "fold_id": fold_id,
        "test_i": test_i,
        "test_j": test_j,
        "actual": actual,
        "predicted": predicted,
    }, metadata


def _build_base_snapshot(raw, method_results, matrix_sha, snapshot_date):
    raw_models = {model["id"]: model for model in raw["models"]}
    raw_benchmarks = {
        benchmark["id"]: benchmark for benchmark in raw["benchmarks"]}
    model_index = {model_id: i for i, model_id in enumerate(MODEL_IDS)}
    benchmark_index = {
        benchmark_id: j for j, benchmark_id in enumerate(BENCH_IDS)}

    models = []
    for model_id in MODEL_IDS:
        source = raw_models[model_id]
        models.append({
            "id": model_id,
            "name": source.get("name"),
            "provider": source.get("provider"),
            "reasoning": source.get("is_reasoning"),
            "release_date": source.get("release_date"),
            "canonical_setting": source.get("canonical_setting"),
            "open_weights": source.get("open_weights"),
        })

    benchmarks = []
    for j, benchmark_id in enumerate(BENCH_IDS):
        source = raw_benchmarks[benchmark_id]
        benchmarks.append({
            "id": benchmark_id,
            "name": source.get("name"),
            "category": source.get("category"),
            "metric": _benchmark_metric(source, M_FULL[:, j]),
            "source_url": source.get("source_url"),
            "canonical_setting": source.get("canonical_setting"),
        })

    sources = [
        [None for _ in BENCH_IDS]
        for _ in MODEL_IDS
    ]
    for score in raw["scores"]:
        i = model_index.get(score["model_id"])
        j = benchmark_index.get(score["benchmark_id"])
        if i is not None and j is not None:
            sources[i][j] = _source_payload(score)
    for score in _LEGACY_SCORES:
        i = model_index.get(score["model_id"])
        j = benchmark_index.get(score["benchmark_id"])
        if i is None or j is None or score.get("score") is None:
            continue
        if float(score["score"]) == float(M_FULL[i, j]):
            sources[i][j] = _source_payload(score)

    predictions = predict_benchpress_scores(
        M_FULL.copy(), metric=BENCH_METRICS, benchmark_ids=BENCH_IDS)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Default full-matrix prediction contains non-finite cells.")
    if not np.array_equal(
            predictions[np.isfinite(M_FULL)], M_FULL[np.isfinite(M_FULL)]):
        raise ValueError("Default predictor changed observed scores.")

    selected = method_results["logit"]["Bias ALS"]
    expected_setting = {"rank": 2, "lam": 0.1}
    if selected.get("best_hp") != expected_setting:
        raise ValueError(
            f"Unexpected default predictor setting: {selected.get('best_hp')}")
    return {
        "models": models,
        "benchmarks": benchmarks,
        "observed": _matrix_to_json(M_FULL),
        "predictions": _matrix_to_json(predictions, digits=6),
        "sources": sources,
        "meta": {
            "snapshot_date": snapshot_date,
            "matrix_sha256": matrix_sha,
            "matrix_shape": list(M_FULL.shape),
            "benchmark_metric_identity_sha256": (
                benchmark_metric_identity_sha256(BENCH_METRICS, BENCH_IDS)
            ),
            "observed_cells": int(np.isfinite(M_FULL).sum()),
            "prediction_method": "Logit Bias ALS",
            "transform": "logit",
            "method": "Bias ALS",
            "rank": 2,
            "lambda": 0.1,
            "source_experiment": os.path.relpath(
                METHOD_RESULTS_PATH, REPO_ROOT),
            "source_prediction_file": os.path.join(
                "experiments",
                "sec4_building_benchpress",
                "method_comparison",
                selected["prediction_file"],
            ),
            "heldout_medape": round(float(selected["medape_median"]), 6),
            "heldout_medae": round(float(selected["medae_median"]), 6),
            "coverage": float(selected["coverage"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--confidence-scores", default=CONFIDENCE_SCORES_PATH)
    parser.add_argument("--confidence-results", default=CONFIDENCE_RESULTS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validation-output")
    args = parser.parse_args()

    with open(MATRIX_PATH) as file:
        raw = json.load(file)
    with open(METHOD_RESULTS_PATH) as file:
        method_results = json.load(file)
    with np.load(args.confidence_scores, allow_pickle=False) as cache:
        confidence_scores = {key: cache[key] for key in cache.files}
    with open(args.confidence_results) as file:
        confidence_results = json.load(file)

    matrix_sha = _sha256(MATRIX_PATH)
    selected_prediction = os.path.join(
        METHOD_DIR,
        method_results["logit"]["Bias ALS"]["prediction_file"],
    )
    table_raw_validation, table_raw, table_metadata = _raw_prediction_validation(
        selected_prediction, "Table 4 selected predictor")
    confidence_raw_validation, confidence_raw, confidence_metadata = (
        _raw_prediction_validation(
        args.confidence_scores, "confidence cache")
    )
    aligned_keys = ["fold_id", "test_i", "test_j", "actual", "predicted"]
    cache_alignment = {
        key: bool(np.array_equal(table_raw[key], confidence_raw[key]))
        for key in aligned_keys
    }
    target_metadata_keys = [
        "n_seeds",
        "n_folds",
        "base_seed",
        "matrix_shape",
        "benchmark_metric_identity_sha256",
        "transform",
        "method",
        "hp",
        "shard_id",
        "shard_index",
    ]
    cached_target_metadata = confidence_metadata.get("target_metadata") or {}
    metadata_alignment = {
        key: cached_target_metadata.get(key) == table_metadata.get(key)
        for key in target_metadata_keys
    }
    if not all(cache_alignment.values()) or not all(metadata_alignment.values()):
        raise ValueError(
            "Confidence cache is not aligned with the selected Table 4 predictor.")
    if confidence_metadata.get(
            "matrix_identity_sha256") != matrix_identity_sha256(M_FULL):
        raise ValueError("Confidence cache matrix identity does not match.")

    data = _build_base_snapshot(
        raw, method_results, matrix_sha, args.snapshot_date)
    attach_prediction_intervals(
        data,
        confidence_scores,
        confidence_results,
        os.path.relpath(args.confidence_scores, REPO_ROOT),
        os.path.relpath(args.confidence_results, REPO_ROOT),
    )

    website_observed = np.asarray([
        [np.nan if value is None else float(value) for value in row]
        for row in data["observed"]
    ], dtype=float)
    observed_mask = np.isfinite(M_FULL)
    differences = np.abs(
        website_observed[observed_mask] - M_FULL[observed_mask])
    mismatch = ~np.isclose(
        website_observed[observed_mask],
        M_FULL[observed_mask],
        rtol=0.0,
        atol=0.0,
    )
    source_coverage = sum(
        data["sources"][i][j] is not None
        for i, j in np.argwhere(observed_mask)
    )
    identity_hash = _observed_identity_hash(M_FULL)
    website_identity_hash = _observed_identity_hash(website_observed)
    prediction_range_violations = 0
    interval_range_violations = 0
    for j, benchmark in enumerate(data["benchmarks"]):
        score_range = (benchmark.get("metric") or {}).get("range")
        if (
            not isinstance(score_range, list)
            or len(score_range) != 2
            or score_range[0] is None
            or score_range[1] is None
        ):
            continue
        lower, upper = map(float, score_range)
        for prediction in (
                data["predictions"][i][j] for i in range(len(MODEL_IDS))):
            if prediction is not None and not lower <= prediction <= upper:
                prediction_range_violations += 1
        for interval in (
                data["prediction_intervals"][i][j]
                for i in range(len(MODEL_IDS))):
            if (
                interval is not None
                and (interval[0] < lower or interval[1] > upper)
            ):
                interval_range_violations += 1
    if (
        int(np.sum(mismatch)) != 0
        or identity_hash != website_identity_hash
        or source_coverage != int(np.sum(observed_mask))
        or prediction_range_violations != 0
        or interval_range_violations != 0
    ):
        raise ValueError("Website observed-score identity validation failed.")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    encoded = (
        json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    with open(args.output, "wb") as file:
        file.write(encoded)

    validation = {
        "snapshot_date": args.snapshot_date,
        "source_matrix": os.path.relpath(MATRIX_PATH, REPO_ROOT),
        "source_matrix_sha256": matrix_sha,
        "matrix_shape": list(M_FULL.shape),
        "observed_cells": int(np.sum(observed_mask)),
        "canonical_observed_identity_sha256": identity_hash,
        "website_observed_identity_sha256": website_identity_hash,
        "website_observed_mismatch_count": int(np.sum(mismatch)),
        "website_observed_max_absolute_difference": (
            0.0 if differences.size == 0 else float(np.max(differences))
        ),
        "observed_cells_with_sources": int(source_coverage),
        "prediction_range_violations": prediction_range_violations,
        "prediction_interval_range_violations": interval_range_violations,
        "finite_point_predictions": int(np.isfinite(np.asarray(
            data["predictions"], dtype=float)).sum()),
        "finite_prediction_intervals": sum(
            interval is not None
            for row in data["prediction_intervals"]
            for interval in row
        ),
        "finite_trust_probabilities": sum(
            value is not None
            for row in data["trust_probabilities"]
            for value in row
        ),
        "table4_raw_scores": table_raw_validation,
        "confidence_raw_scores": confidence_raw_validation,
        "confidence_cache_alignment": cache_alignment,
        "confidence_target_metadata_alignment": metadata_alignment,
        "confidence_matrix_identity_sha256": confidence_metadata[
            "matrix_identity_sha256"
        ],
        "confidence_benchmark_metric_identity_sha256": confidence_metadata[
            "benchmark_metric_identity_sha256"
        ],
        "output": os.path.relpath(args.output, REPO_ROOT),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    validation_path = (
        args.validation_output
        or os.path.join(output_dir, "data.validation.json")
    )
    with open(validation_path, "w") as file:
        json.dump(validation, file, indent=2, sort_keys=True)
        file.write("\n")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
