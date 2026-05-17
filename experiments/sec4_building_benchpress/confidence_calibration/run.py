#!/usr/bin/env python3
"""Evaluate confidence calibration for BenchPress predictions.

This script is prediction-cache first. It reads the Section 4.2 held-out
prediction shards, computes uncertainty scores, writes per-cell confidence
outputs, and derives aggregate calibration metrics from that cache.
"""

import argparse
import contextlib
import io
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEC4_DIR = os.path.dirname(SCRIPT_DIR)
METHOD_DIR = os.path.join(SEC4_DIR, "method_comparison")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")
SCORES_PATH = os.path.join(SCRIPT_DIR, "confidence_scores.npz")
TARGET_PREDICTION_REL = "predictions/0154__logit__bias_als__hp04_b16f05a66b.npz"
EXPECTED_METHOD_PROTOCOL = {
    "n_seeds": 10,
    "n_folds": 3,
    "base_seed": 42,
    "matrix_shape": None,
}
EXPECTED_TARGET = {
    "best_hp": {"lam": 0.1, "rank": 2},
    "best_hp_index": 4,
    "prediction_file": TARGET_PREDICTION_REL,
}


with contextlib.redirect_stdout(io.StringIO()):
    from benchpress.evaluation_harness import (
        M_FULL,
        compute_prediction_error,
    )
    from benchpress.io_utils import load_json, write_json_atomic, write_npz_compressed_atomic
    from benchpress.methods.confidence import (
        confidence_feature_sets,
        conformal_interval,
        coverage_width,
        leave_fold_mlp_error_calibrator,
        stack_features,
        structural_support_features,
        summarize_confidence_method,
    )

# Backward-compatible aliases used by plot.py for appendix table recomputation.
_conformal_interval = conformal_interval
_coverage_width = coverage_width


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _metadata(data):
    return json.loads(str(data["metadata_json"]))


def _prediction_path(rel_path):
    return os.path.join(METHOD_DIR, rel_path)


def _target_prediction_info():
    results = load_json(os.path.join(METHOD_DIR, "results.json"))
    row = results["logit"]["Bias ALS"]
    actual = {key: row.get(key) for key in EXPECTED_TARGET}
    if actual != EXPECTED_TARGET:
        raise ValueError(
            "Confidence calibration target predictor changed; expected "
            f"{EXPECTED_TARGET}, found {actual}."
        )
    return _prediction_path(row["prediction_file"]), row


def _shard_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _shard_index(path):
    return int(_shard_id(path).split("__", 1)[0])


def _target_metadata(data, path, row):
    metadata = _metadata(data)
    metadata.update({
        "path": os.path.relpath(path, METHOD_DIR),
        "shard_id": _shard_id(path),
        "shard_index": _shard_index(path),
        "transform": "logit",
        "method": "Bias ALS",
        "hp": row["best_hp"],
        "hp_index": row["best_hp_index"],
    })
    return metadata


def _manifest_rows():
    manifest = load_json(os.path.join(METHOD_DIR, "manifest.json"))
    if int(manifest.get("n_missing_shards", -1)) != 0:
        raise ValueError("Method-comparison manifest has missing shards")
    return manifest["completed"]


def _method_results_rows():
    results = load_json(os.path.join(METHOD_DIR, "results.json"))
    rows = []
    for transform, methods in results.items():
        for method, payload in methods.items():
            rows.append({
                "transform": transform,
                "method": method,
                **payload,
            })
    return rows


def _ensemble_prediction_files(ensemble_transform):
    files = []
    for row in _manifest_rows():
        if row["method"] != "Bias ALS":
            continue
        if ensemble_transform != "all" and row["transform"] != ensemble_transform:
            continue
        files.append(_prediction_path(row["prediction_file"]))
    if not files:
        raise ValueError(
            f"No Bias ALS ensemble shards found for transform={ensemble_transform!r}")
    return sorted(files)


def _strong_method_prediction_files(max_methods=12):
    """Top complete methods by Section 4.2 MedAPE, excluding the target itself."""
    files = []
    rows = sorted(_method_results_rows(), key=lambda r: r["medape_median"])
    for row in rows:
        if row["coverage"] < 0.999:
            continue
        if row["transform"] == "logit" and row["method"] == "Bias ALS":
            continue
        files.append(_prediction_path(row["prediction_file"]))
        if len(files) >= max_methods:
            break
    if len(files) < 3:
        raise ValueError("Need at least three strong-method prediction files")
    return files


def _assert_aligned(reference, candidate, candidate_path):
    keys = ["fold_id", "test_i", "test_j", "actual"]
    for key in keys:
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Prediction shard is not aligned on {key}: {candidate_path}")


def _assert_prediction_cache_metadata(data, path):
    metadata = _metadata(data)
    expected = dict(EXPECTED_METHOD_PROTOCOL)
    expected["matrix_shape"] = list(M_FULL.shape)
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Prediction cache protocol mismatch in {path}: "
            f"expected {expected}, found {actual}"
        )
    if metadata.get("path") != os.path.relpath(path, METHOD_DIR):
        raise ValueError(
            f"Prediction cache path metadata mismatch in {path}: "
            f"found {metadata.get('path')}"
        )


def _stack_from_files(reference, files):
    predictions = []
    for path in files:
        candidate = _load_npz(path)
        _assert_prediction_cache_metadata(candidate, path)
        _assert_aligned(reference, candidate, path)
        predictions.append(candidate["predicted"].astype(float))
    return np.vstack(predictions)


def build_confidence_scores(ensemble_transform="logit",
                            max_strong_methods=12,
                            risk_methods=None,
                            fold_shard_index=None,
                            num_fold_shards=None,
                            scores_path=SCORES_PATH):
    if risk_methods is None:
        risk_methods = ["disagreement", "structural_support", "combined_risk_model"]
    risk_methods = set(risk_methods)
    target_path, target_row = _target_prediction_info()
    target = _load_npz(target_path)
    _assert_prediction_cache_metadata(target, target_path)
    target_meta = _target_metadata(target, target_path, target_row)

    hp_files = _ensemble_prediction_files(ensemble_transform)
    hp_stack = _stack_from_files(target, hp_files)
    hp_features = stack_features(hp_stack, target["predicted"].astype(float))

    strong_files = _strong_method_prediction_files(max_methods=max_strong_methods)
    strong_stack = _stack_from_files(target, strong_files)
    strong_features = stack_features(strong_stack, target["predicted"].astype(float))
    structural_features = structural_support_features(target)

    arrays = {
        "fold_id": target["fold_id"].astype(np.int16),
        "test_i": target["test_i"].astype(np.int16),
        "test_j": target["test_j"].astype(np.int16),
        "actual": target["actual"].astype(float),
        "predicted": target["predicted"].astype(float),
        "bias_als_hp_disagreement_uncertainty": hp_features["mad"].astype(float),
        "strong_method_disagreement_uncertainty": strong_features["mad"].astype(float),
    }

    feature_sets = confidence_feature_sets(
        hp_features, strong_features, structural_features)
    disagreement_features = feature_sets["disagreement"]
    combined_features = feature_sets["combined_risk_model"]

    all_folds = np.unique(arrays["fold_id"].astype(int))
    folds_to_run = all_folds
    if fold_shard_index is not None or num_fold_shards is not None:
        if fold_shard_index is None or num_fold_shards is None:
            raise ValueError("fold_shard_index and num_fold_shards must be set together")
        folds_to_run = all_folds[int(fold_shard_index)::int(num_fold_shards)]
        print(f"Running fold shard {fold_shard_index}/{num_fold_shards}: "
              f"{[int(f) for f in folds_to_run]}", flush=True)

    disagreement_feature_names = sorted(disagreement_features)
    structural_feature_names = sorted(structural_features)
    combined_feature_names = sorted(combined_features)
    selected_by_method = {}

    if "disagreement" in risk_methods:
        disagreement_uncertainty, disagreement_feature_names, selected = (
            leave_fold_mlp_error_calibrator(
                arrays["actual"], arrays["predicted"], arrays["fold_id"],
                disagreement_features, folds_to_run=folds_to_run,
                label="disagreement")
        )
        arrays["disagreement_uncertainty"] = disagreement_uncertainty.astype(float)
        selected_by_method["disagreement"] = selected

    if "structural_support" in risk_methods:
        structural_uncertainty, structural_feature_names, selected = (
            leave_fold_mlp_error_calibrator(
                arrays["actual"], arrays["predicted"], arrays["fold_id"],
                structural_features, folds_to_run=folds_to_run,
                label="structural_support")
        )
        arrays["structural_support_uncertainty"] = structural_uncertainty.astype(float)
        selected_by_method["structural_support"] = selected

    if "combined_risk_model" in risk_methods:
        combined_uncertainty, combined_feature_names, selected = (
            leave_fold_mlp_error_calibrator(
                arrays["actual"], arrays["predicted"], arrays["fold_id"],
                combined_features, folds_to_run=folds_to_run,
                label="combined_risk_model")
        )
        arrays["combined_risk_model_uncertainty"] = combined_uncertainty.astype(float)
        selected_by_method["combined_risk_model"] = selected

    metadata = {
        "target_prediction_file": os.path.relpath(target_path, SCRIPT_DIR),
        "target_metadata": target_meta,
        "ensemble_transform": ensemble_transform,
        "bias_als_hp_files": [os.path.relpath(p, SCRIPT_DIR) for p in hp_files],
        "strong_method_files": [os.path.relpath(p, SCRIPT_DIR) for p in strong_files],
        "disagreement_features": disagreement_feature_names,
        "structural_support_features": structural_feature_names,
        "combined_risk_model_features": combined_feature_names,
        "mlp_hidden_grid": [[16], [32], [64, 32]],
        "mlp_selected_hidden_layers_by_fold": selected_by_method,
        "risk_methods": sorted(risk_methods),
        "folds_run": [int(f) for f in folds_to_run],
        "matrix_shape": list(M_FULL.shape),
        "base_seed": SEED,
    }

    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    write_npz_compressed_atomic(scores_path, **arrays)
    return arrays, metadata


def summarize(arrays, metadata, scores_path=SCORES_PATH, results_path=RESULTS_PATH):
    actual = arrays["actual"]
    predicted = arrays["predicted"]
    fold_id = arrays["fold_id"]

    methods = {}
    for key in arrays:
        if not key.endswith("_uncertainty"):
            continue
        name = key[:-len("_uncertainty")]
        lower = arrays.get(f"{name}_lower")
        upper = arrays.get(f"{name}_upper")
        method_name, payload = summarize_confidence_method(
            name, actual, predicted, fold_id, arrays[key], lower=lower, upper=upper)
        methods[method_name] = payload

    point_metrics = compute_prediction_error(actual, predicted)

    results = {
        "setting": {
            "point_predictor": "BenchPress = Logit Bias ALS, rank=2, lam=0.1",
            "evaluation": "same held-out folds as Section 4.2 method comparison",
            "n_test_predictions": int(len(actual)),
            "matrix_shape": metadata["matrix_shape"],
        },
        "point_prediction_metrics": {
            "medape": point_metrics["medape"],
            "medae": point_metrics["medae"],
        },
        "confidence_methods": methods,
        "cache": {
            "confidence_scores": os.path.relpath(scores_path, SCRIPT_DIR),
            **metadata,
        },
    }
    write_json_atomic(results_path, results, indent=2, sort_keys=True)
    return results


def merge_score_shards(paths, scores_path=SCORES_PATH, results_path=RESULTS_PATH):
    if not paths:
        raise ValueError("No score shards provided")
    shards = [_load_npz(path) for path in paths]
    base_keys = [
        "fold_id", "test_i", "test_j", "actual", "predicted",
        "bias_als_hp_disagreement_uncertainty",
        "strong_method_disagreement_uncertainty",
    ]
    arrays = {key: shards[0][key] for key in base_keys}
    metadata = {}
    selected = {}
    for shard in shards:
        meta = _metadata(shard)
        for key, value in meta.items():
            if key == "mlp_selected_hidden_layers_by_fold":
                for method, folds in value.items():
                    selected.setdefault(method, {}).update(folds)
            elif key in {"risk_methods", "folds_run"}:
                continue
            else:
                metadata[key] = value
        for key, value in shard.items():
            if not key.endswith("_uncertainty") or key in base_keys:
                continue
            value = value.astype(float)
            if key not in arrays:
                arrays[key] = np.full_like(value, np.nan, dtype=float)
            valid = np.isfinite(value)
            arrays[key][valid] = value[valid]
    metadata["mlp_selected_hidden_layers_by_fold"] = selected
    metadata["risk_methods"] = sorted(
        key[:-len("_uncertainty")]
        for key in arrays
        if key.endswith("_uncertainty") and key not in base_keys
    )
    missing = {
        key: int(np.sum(~np.isfinite(value)))
        for key, value in arrays.items()
        if key.endswith("_uncertainty") and key not in base_keys
    }
    if any(count > 0 for count in missing.values()):
        raise ValueError(f"Merged shards still have missing uncertainties: {missing}")
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    write_npz_compressed_atomic(scores_path, **arrays)
    return summarize(arrays, metadata, scores_path=scores_path, results_path=results_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-transform", default="logit",
                        choices=["probit", "logit", "identity", "log", "asinh",
                                 "sqrt", "quantile", "all"])
    parser.add_argument("--max-strong-methods", type=int, default=12)
    parser.add_argument("--risk-methods", nargs="+",
                        choices=["disagreement", "structural_support",
                                 "combined_risk_model"],
                        default=["disagreement", "structural_support",
                                 "combined_risk_model"])
    parser.add_argument("--fold-shard-index", type=int)
    parser.add_argument("--num-fold-shards", type=int)
    parser.add_argument("--scores-path", default=SCORES_PATH)
    parser.add_argument("--results-path", default=RESULTS_PATH)
    parser.add_argument("--skip-results", action="store_true")
    parser.add_argument("--merge-scores", nargs="+")
    args = parser.parse_args()

    if args.merge_scores:
        results = merge_score_shards(
            args.merge_scores, scores_path=args.scores_path,
            results_path=args.results_path)
        print(f"Wrote {args.scores_path}")
        print(f"Wrote {args.results_path}")
        print(json.dumps(results["point_prediction_metrics"], indent=2, sort_keys=True))
        return

    arrays, metadata = build_confidence_scores(
        ensemble_transform=args.ensemble_transform,
        max_strong_methods=args.max_strong_methods,
        risk_methods=args.risk_methods,
        fold_shard_index=args.fold_shard_index,
        num_fold_shards=args.num_fold_shards,
        scores_path=args.scores_path)
    results = None
    if not args.skip_results:
        results = summarize(
            arrays, metadata, scores_path=args.scores_path,
            results_path=args.results_path)

    print(f"Wrote {args.scores_path}")
    if results is not None:
        print(f"Wrote {args.results_path}")
        print(json.dumps(results["point_prediction_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
