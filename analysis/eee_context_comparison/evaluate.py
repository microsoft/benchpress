"""Matched-fold prediction, confidence, and redundancy evaluation."""

import json
import os
import platform
import subprocess
import sys

import numpy as np

from benchpress.evaluation_harness import (
    BENCH_IDS,
    M_FULL,
    compute_prediction_error,
    holdout_per_model,
    mask_cells,
    mask_columns,
)
from benchpress.io_utils import load_json, write_json_atomic
from benchpress.methods.confidence import (
    conformal_interval,
    coverage_width,
    leave_fold_mlp_error_calibrator,
    structural_support_features_for_cells,
)
from benchpress.methods.predictors import predict_benchpress_scores
from benchpress.stats import paired_wilcoxon, wilcoxon_grouped_median

from common import (
    MIN_BENCHMARKS_PER_MODEL,
    MIN_MODELS_PER_BENCHMARK,
    build_matched_targets,
    canonical_benchpress_json_path,
    file_sha256,
)


BASE_SEED = 42
N_FOLDS = 3
REDUNDANCY_CORRELATION = 0.98
REDUNDANCY_MIN_OVERLAP = 8


def mapped_fold_cells(fold_cells, target_metadata, condition):
    """Translate compressed matched-target coordinates to a full matrix."""
    field = "benchpress_cell" if condition == "benchpress" else "eee_cell"
    return [
        tuple(target_metadata[(int(i), int(j))][field])
        for i, j in fold_cells
    ]


def near_duplicate_columns(M_train, benchmark_ids, target_columns,
                           protected_columns):
    """Find training-only near-duplicate columns for the EEE ablation."""
    removed = {}
    protected = set(int(value) for value in protected_columns)
    for target_j in sorted(set(int(value) for value in target_columns)):
        target_values = M_train[:, target_j]
        matches = []
        for candidate_j in range(M_train.shape[1]):
            if candidate_j == target_j or candidate_j in protected:
                continue
            candidate_values = M_train[:, candidate_j]
            shared = np.isfinite(target_values) & np.isfinite(candidate_values)
            overlap = int(shared.sum())
            if overlap < REDUNDANCY_MIN_OVERLAP:
                continue
            target_shared = target_values[shared]
            candidate_shared = candidate_values[shared]
            if (np.std(target_shared) <= 1e-12
                    or np.std(candidate_shared) <= 1e-12):
                continue
            correlation = float(np.corrcoef(
                target_shared, candidate_shared)[0, 1])
            if (np.isfinite(correlation)
                    and abs(correlation) > REDUNDANCY_CORRELATION):
                matches.append({
                    "column_index": int(candidate_j),
                    "benchmark_id": benchmark_ids[candidate_j],
                    "correlation": correlation,
                    "overlap": overlap,
                })
        if matches:
            removed[str(target_j)] = matches
    return removed


def run_fold(condition, fold_id, fold_cells, target_metadata, eee):
    """Predict one shared fold under one matrix-context condition."""
    matrix_condition = "eee" if condition.startswith("eee") else "benchpress"
    full_cells = mapped_fold_cells(
        fold_cells, target_metadata, matrix_condition)
    if matrix_condition == "benchpress":
        M_train = mask_cells(full_cells, base_matrix=M_FULL)
        benchmark_ids = BENCH_IDS
        metric = None
        removed = {}
    else:
        M_train = mask_cells(full_cells, base_matrix=eee["values"])
        benchmark_ids = eee["benchmark_ids"]
        metric = eee["metric"]
        removed = {}
        if condition == "eee_deduplicated":
            target_columns = [j for _, j in full_cells]
            protected_columns = [
                metadata["eee_cell"][1]
                for metadata in target_metadata.values()
            ]
            removed = near_duplicate_columns(
                M_train, benchmark_ids, target_columns, protected_columns)
            drop_columns = sorted({
                row["column_index"]
                for matches in removed.values()
                for row in matches
            })
            M_train = mask_columns(drop_columns, base_matrix=M_train)

    predicted_matrix = predict_benchpress_scores(
        M_train, metric=metric, benchmark_ids=benchmark_ids)
    features = (
        structural_support_features_for_cells(M_train, full_cells)
        if condition in {"benchpress", "eee"}
        else None
    )
    records = []
    for local_idx, ((mapped_i, mapped_j), (full_i, full_j)) in enumerate(
            zip(fold_cells, full_cells)):
        metadata = target_metadata[(int(mapped_i), int(mapped_j))]
        record = {
            **metadata,
            "condition": condition,
            "fold_id": int(fold_id),
            "predicted": float(predicted_matrix[full_i, full_j]),
        }
        if features is not None:
            record["structural_features"] = {
                key: float(values[local_idx])
                for key, values in features.items()
            }
        records.append(record)
    return {
        "condition": condition,
        "fold_id": int(fold_id),
        "records": records,
        "redundancy_removed": removed,
    }


def validate_fold_payload(payload, condition, fold_id, target_keys):
    """Reject stale or incomplete resumable fold artifacts."""
    if payload.get("condition") != condition:
        raise ValueError(f"stale fold condition in {condition} fold {fold_id}")
    if int(payload.get("fold_id", -1)) != int(fold_id):
        raise ValueError(f"stale fold id in {condition} fold {fold_id}")
    observed_keys = {row["target_key"] for row in payload.get("records", [])}
    if observed_keys != set(target_keys):
        raise ValueError(f"stale target set in {condition} fold {fold_id}")


def condition_metrics(records):
    """Compute shared error metrics from raw per-cell predictions."""
    actual = np.asarray([row["actual"] for row in records], dtype=float)
    predicted = np.asarray([row["predicted"] for row in records], dtype=float)
    fold_ids = np.asarray([row["fold_id"] for row in records], dtype=int)
    pooled = compute_prediction_error(actual, predicted)
    grouped = compute_prediction_error(
        actual, predicted, groups=fold_ids, aggregation="per_group_median")
    return {
        "pooled": pooled,
        "per_fold_median": {
            "n_folds": int(grouped["n_groups"]),
            "medae": float(grouped["medae_median"]),
            "medape": float(grouped["medape_median"]),
        },
    }


def confidence_metrics(records, label):
    """Cross-fit the matrix-support risk model and calibrate 90% intervals."""
    actual = np.asarray([row["actual"] for row in records], dtype=float)
    predicted = np.asarray([row["predicted"] for row in records], dtype=float)
    fold_ids = np.asarray([row["fold_id"] for row in records], dtype=int)
    feature_names = sorted(records[0]["structural_features"])
    features = {
        name: np.asarray([
            row["structural_features"][name] for row in records
        ], dtype=float)
        for name in feature_names
    }
    uncertainty, selected_features, selected_models = (
        leave_fold_mlp_error_calibrator(
            actual,
            predicted,
            fold_ids,
            features,
            label=label,
            seed=BASE_SEED,
        )
    )
    lower, upper, scale = conformal_interval(
        actual, predicted, uncertainty, fold_ids, ci=0.90)
    metrics = coverage_width(actual, lower, upper)
    metrics.update({
        "feature_names": selected_features,
        "selected_risk_models": selected_models,
        "median_uncertainty": float(np.nanmedian(uncertainty)),
        "median_conformal_scale": float(np.nanmedian(scale)),
    })
    intervals = {
        row["target_key"]: {
            "lower": float(lo),
            "upper": float(hi),
            "width": float(hi - lo),
            "uncertainty": float(unc),
        }
        for row, lo, hi, unc in zip(records, lower, upper, uncertainty)
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(unc)
    }
    return metrics, intervals


def paired_accuracy(merged):
    """Compare absolute errors on the identical ordered target cells."""
    ordered = {
        condition: {row["target_key"]: row for row in records}
        for condition, records in merged.items()
    }
    shared_keys = sorted(set.intersection(
        *(set(records) for records in ordered.values())))
    errors = {
        condition: np.asarray([
            abs(records[key]["predicted"] - records[key]["actual"])
            for key in shared_keys
        ])
        for condition, records in ordered.items()
    }
    eee_delta = errors["eee"] - errors["benchpress"]
    dedup_delta = errors["eee_deduplicated"] - errors["eee"]
    delta_records = [
        {
            "benchmark_id":
                ordered["benchpress"][key]["benchpress_benchmark_id"],
            "eee_minus_benchpress_abs_error": float(
                errors["eee"][idx] - errors["benchpress"][idx]),
            "deduplicated_minus_full_eee_abs_error": float(
                errors["eee_deduplicated"][idx] - errors["eee"][idx]),
        }
        for idx, key in enumerate(shared_keys)
    ]
    per_benchmark_test = wilcoxon_grouped_median(
        delta_records,
        [
            "eee_minus_benchpress_abs_error",
            "deduplicated_minus_full_eee_abs_error",
        ],
        group_key="benchmark_id",
        delta_key_template="{metric}",
        min_groups=5,
        include_sign_counts=True,
    )
    by_benchmark = {}
    for benchmark_id in sorted({
            row["benchmark_id"] for row in delta_records}):
        rows = [
            row for row in delta_records
            if row["benchmark_id"] == benchmark_id
        ]
        by_benchmark[benchmark_id] = {
            "n": int(len(rows)),
            "median_eee_minus_benchpress_abs_error": float(np.median([
                row["eee_minus_benchpress_abs_error"] for row in rows
            ])),
            "median_deduplicated_minus_full_eee_abs_error": float(np.median([
                row["deduplicated_minus_full_eee_abs_error"] for row in rows
            ])),
        }
    _, eee_cell_p = paired_wilcoxon(
        errors["eee"], errors["benchpress"])
    _, dedup_cell_p = paired_wilcoxon(
        errors["eee_deduplicated"], errors["eee"])
    return {
        "eee_minus_benchpress_abs_error": {
            "median": float(np.median(eee_delta)),
            "eee_lower_fraction": float(np.mean(eee_delta < 0)),
            "ties_fraction": float(np.mean(np.isclose(eee_delta, 0))),
            "cell_level_p_value_diagnostic": float(eee_cell_p),
            "n": int(len(shared_keys)),
        },
        "deduplicated_minus_full_eee_abs_error": {
            "median": float(np.median(dedup_delta)),
            "deduplicated_lower_fraction": float(np.mean(dedup_delta < 0)),
            "ties_fraction": float(np.mean(np.isclose(dedup_delta, 0))),
            "cell_level_p_value_diagnostic": float(dedup_cell_p),
            "n": int(len(shared_keys)),
        },
        "per_benchmark_wilcoxon": per_benchmark_test,
        "by_benchmark": by_benchmark,
    }


def summarize(merged, target_metadata):
    """Build the final matched accuracy, interval, and ablation report."""
    accuracy = {
        condition: condition_metrics(records)
        for condition, records in merged.items()
    }
    benchpress_confidence, bp_intervals = confidence_metrics(
        merged["benchpress"], "benchpress_structural_support")
    eee_confidence, eee_intervals = confidence_metrics(
        merged["eee"], "eee_structural_support")
    shared_interval_keys = sorted(set(bp_intervals) & set(eee_intervals))
    width_differences = np.asarray([
        eee_intervals[key]["width"] - bp_intervals[key]["width"]
        for key in shared_interval_keys
    ], dtype=float)
    _, width_p = paired_wilcoxon(
        [eee_intervals[key]["width"] for key in shared_interval_keys],
        [bp_intervals[key]["width"] for key in shared_interval_keys],
    )

    unique_targets = list(target_metadata.values())
    actual = np.asarray([row["actual"] for row in unique_targets], dtype=float)
    eee_observed = np.asarray([
        row["eee_observed_score"] for row in unique_targets
    ], dtype=float)
    score_disagreement = np.abs(eee_observed - actual)
    return {
        "target_set": {
            "matched_cells": int(len(unique_targets)),
            "models": int(len({
                row["benchpress_model_id"] for row in unique_targets
            })),
            "benchmarks": int(len({
                row["benchpress_benchmark_id"] for row in unique_targets
            })),
        },
        "direct_eee_score_alignment_to_benchpress_ground_truth": {
            **compute_prediction_error(actual, eee_observed),
            "within_1_point_fraction": float(np.mean(score_disagreement <= 1)),
            "within_5_points_fraction": float(np.mean(score_disagreement <= 5)),
            "within_10_points_fraction": float(np.mean(score_disagreement <= 10)),
        },
        "accuracy": accuracy,
        "paired_accuracy": paired_accuracy(merged),
        "confidence_90": {
            "benchpress": benchpress_confidence,
            "eee": eee_confidence,
            "paired_eee_minus_benchpress_width": {
                "median": float(np.median(width_differences)),
                "eee_narrower_fraction": float(np.mean(width_differences < 0)),
                "cell_level_p_value_diagnostic": float(width_p),
                "n": int(len(width_differences)),
            },
        },
    }


def git_commit():
    """Return the exact source commit for the run manifest."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def build_manifest(eee, mapping, mapping_path, target_metadata):
    """Build the immutable run identity and protocol manifest."""
    bp_json_path = canonical_benchpress_json_path()
    return {
        "source_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": {
            "seed": BASE_SEED,
            "n_folds": N_FOLDS,
            "predictor": "logit_bias_als_rank2_lambda0.1",
            "confidence": "crossfit_matrix_support_leave_fold_conformal_90",
            "eee_filter": {
                "percentage_columns_only": True,
                "min_benchmarks_per_model": MIN_BENCHMARKS_PER_MODEL,
                "min_models_per_benchmark": MIN_MODELS_PER_BENCHMARK,
            },
            "redundancy": {
                "absolute_correlation_gt": REDUNDANCY_CORRELATION,
                "min_shared_models": REDUNDANCY_MIN_OVERLAP,
                "training_only": True,
            },
        },
        "inputs": {
            "benchpress_json": {
                "path": bp_json_path,
                "sha256": file_sha256(bp_json_path),
            },
            "eee_scores_csv": {
                "path": eee["scores_path"],
                "sha256": file_sha256(eee["scores_path"]),
            },
            "mapping": {
                "path": os.path.abspath(mapping_path),
                "sha256": file_sha256(mapping_path),
            },
        },
        "matrices": {
            "benchpress": {
                "shape": [int(M_FULL.shape[0]), int(M_FULL.shape[1])],
                "observed_cells": int(np.isfinite(M_FULL).sum()),
            },
            "eee": {
                "shape": [
                    int(eee["values"].shape[0]),
                    int(eee["values"].shape[1]),
                ],
                "observed_cells": int(np.isfinite(eee["values"]).sum()),
            },
        },
        "targets": {
            "matched_cells": int(len(target_metadata)),
            "mapping": mapping,
        },
    }


def run_evaluation(eee, mapping_path, output):
    """Run or resume all shared folds and write merged results."""
    mapping = load_json(mapping_path)
    target_matrix, target_metadata = build_matched_targets(eee, mapping)
    folds = holdout_per_model(
        min_scores=1,
        n_folds=N_FOLDS,
        seed=BASE_SEED,
        M=target_matrix,
    )
    os.makedirs(os.path.join(output, "folds"), exist_ok=True)

    manifest = build_manifest(eee, mapping, mapping_path, target_metadata)
    manifest_path = os.path.join(output, "manifest.json")
    if os.path.exists(manifest_path):
        if load_json(manifest_path) != manifest:
            raise RuntimeError(
                "output manifest differs from this run; use an empty output directory")
    else:
        write_json_atomic(
            manifest_path, manifest, indent=2, sort_keys=True,
            trailing_newline=True)

    conditions = ["benchpress", "eee", "eee_deduplicated"]
    merged = {condition: [] for condition in conditions}
    redundancy = {}
    for fold_id, (_, fold_cells) in enumerate(folds):
        fold_keys = [
            target_metadata[(int(i), int(j))]["target_key"]
            for i, j in fold_cells
        ]
        for condition in conditions:
            path = os.path.join(
                output, "folds", f"{condition}_fold_{fold_id}.json")
            if os.path.exists(path):
                payload = load_json(path)
                validate_fold_payload(
                    payload, condition, fold_id, fold_keys)
                print(f"reused {path}", flush=True)
            else:
                print(
                    f"[{condition}] fold {fold_id} "
                    f"targets={len(fold_cells)} start",
                    flush=True,
                )
                payload = run_fold(
                    condition, fold_id, fold_cells, target_metadata, eee)
                write_json_atomic(
                    path, payload, indent=2, sort_keys=True,
                    trailing_newline=True)
                print(f"[{condition}] fold {fold_id} done", flush=True)
            merged[condition].extend(payload["records"])
            if condition == "eee_deduplicated":
                redundancy[str(fold_id)] = payload["redundancy_removed"]

    write_json_atomic(
        os.path.join(output, "records.json"),
        merged,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )
    results = summarize(merged, target_metadata)
    results["redundancy_removed_by_fold"] = redundancy
    write_json_atomic(
        os.path.join(output, "results.json"),
        results,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
