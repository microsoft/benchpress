"""Reproduce the public quantization transfer comparison with clean lookups.

This diagnostic does not change the BenchPress predictor or the paper.
Run on a remote CPU host with the installed benchpress environment:

    python analysis/quantization_transfer.py --stage sanity \
        --output-root /tmp/benchpress-quantization-transfer
    python analysis/quantization_transfer.py --stage transfer \
        --output-root /tmp/benchpress-quantization-transfer --workers 8

The transfer protocol matches quant-delta-predictor/src/track2_benchpress.py:
acc_before >= 20, base and quantized rows, three revealed scores, seeds 0..4,
and all other rows visible. Lookup variants exclude the hidden target cells,
the target checkpoint, or its entire base-model group. The original all-data
lookup is retained only to reproduce the published comparison.

Each target is saved atomically with raw scores, both the matrix-averaged and
card-specific base scores, lookup training counts, and the revealed columns.
Existing units must match the config and target specification before reuse.
The first target runs serially before the remaining targets run in parallel.
The sanity stage validates and reuses the existing 10-seed/3-fold raw paper
predictions, then checks the current predictor against the first cached fold.
No canonical prediction artifact is replaced. Copy the output root to durable
project storage after running; the remote sandbox is not permanent storage.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from urllib.request import urlopen

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_METRICS,
    FOLDS_DIR,
    M_FULL,
    benchmark_metric_identity_sha256,
    compute_prediction_error,
    load_folds,
    mask_cells,
    matrix_identity_sha256,
)
from benchpress.io_utils import load_json, write_json_atomic
from benchpress.methods.predictors import predict_benchpress_scores
from benchpress.shard_utils import canonical_hp_json, hp_short_hash


SOURCE_COMMIT = "87b699afb2d93d65809d298d15b78d4037b0c8b3"
DATA_BLOB = "7219c130887abab69a98a7824cfaf2bc8be4dc82"
DATA_URL = (
    "https://raw.githubusercontent.com/gracejackson-sudo/"
    f"quant-delta-predictor/{SOURCE_COMMIT}/data/dataset.csv"
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PREDICTIONS = os.path.join(
    REPO_ROOT, "experiments", "sec4_building_benchpress", "method_comparison",
    "predictions", "0124__logit__bias_als__hp01_b16f05a66b.npz")


def score_records(records, actual_key, prediction_keys):
    """Summarize matched raw predictions in points, without percentage errors."""
    actual = np.asarray([row[actual_key] for row in records])
    result = {}
    for key in prediction_keys:
        predicted = np.asarray([row[key] for row in records])
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError(f"Nonfinite values in {actual_key}/{key}")
        metrics = compute_prediction_error(actual, predicted)
        result[key] = {
            "n": metrics["n"],
            "mae": float(mean_absolute_error(actual, predicted)),
            "medae": metrics["medae"],
        }
    return result


def run_sanity_fold(job, output_dir, config):
    """Validate a cached fold; freshly compare the first fold to current code."""
    fold, matrix, cells, cached = job
    path = os.path.join(output_dir, f"fold_{fold:02d}.json")
    specification = {"fold": fold, "cells": [list(cell) for cell in cells]}
    if os.path.exists(path):
        payload = load_json(path)
        if payload["config"] != config or payload["specification"] != specification:
            raise ValueError(f"Stale fold artifact: {path}")
        if [(row["i"], row["j"]) for row in payload["records"]] != cells:
            raise ValueError(f"Incomplete fold artifact: {path}")
        return payload["records"]
    observed = np.isfinite(matrix)
    np.testing.assert_array_equal(cached[observed], matrix[observed])
    if fold == 0:
        with threadpool_limits(limits=1):
            predicted = predict_benchpress_scores(matrix)
        np.testing.assert_allclose(predicted, cached, rtol=0, atol=1e-9)
    records = [
        {"fold": fold, "i": int(i), "j": int(j),
         "actual": float(M_FULL[i, j]), "predicted": float(cached[i, j]),
         **({"current_predicted": float(predicted[i, j])} if fold == 0 else {})}
        for i, j in cells
    ]
    score_records(records, "actual", ["predicted"])
    write_json_atomic(path, {
        "config": config, "specification": specification, "records": records,
    })
    return records


def run_transfer_target(job, data, matrix, output_dir, config):
    """Predict one masked quantized row and score leakage-controlled lookups."""
    seed, model, revealed, hidden = job
    specification = {
        "seed": seed, "model": model, "revealed": revealed, "hidden": hidden,
    }
    path = os.path.join(output_dir, f"target_{hp_short_hash(specification)}.json")
    if os.path.exists(path):
        payload = load_json(path)
        if payload["config"] != config or payload["specification"] != specification:
            raise ValueError(f"Stale target artifact: {path}")
        if [row["benchmark"] for row in payload["records"]] != hidden:
            raise ValueError(f"Incomplete target artifact: {path}")
        return payload["records"]

    own = data.loc[data.model == model]
    if own.base_model.nunique() != 1 or own.scheme.nunique() != 1:
        raise ValueError(f"Ambiguous base model or scheme: {model}")
    base_model, scheme = own.base_model.iloc[0], own.scheme.iloc[0]
    row = matrix.index.get_loc("QUANT::" + model)
    base_row = matrix.index.get_loc("BASE::" + base_model)
    cells = [(row, matrix.columns.get_loc(benchmark)) for benchmark in hidden]
    masked = mask_cells(cells, base_matrix=matrix.to_numpy(float))
    if int(np.isfinite(masked[row]).sum()) != len(revealed):
        raise ValueError(f"Unexpected visible target cells: {model}")
    metric = {name: {"type": "pct", "range": [0.0, 100.0]}
              for name in matrix.columns}
    with threadpool_limits(limits=1):
        predicted = predict_benchpress_scores(
            masked, metric=metric, benchmark_ids=list(matrix.columns))
    np.testing.assert_array_equal(predicted[np.isfinite(masked)],
                                  masked[np.isfinite(masked)])

    hidden_rows = (data.model == model) & data.benchmark.isin(hidden)
    eligible = {
        "lookup_all": data.scheme == scheme,
        "lookup_hidden_excluded": (data.scheme == scheme) & ~hidden_rows,
        "lookup_checkpoint_excluded": (data.scheme == scheme) & (data.model != model),
        "lookup_base_excluded": (data.scheme == scheme) & (data.base_model != base_model),
    }
    lookups, counts = {}, {}
    for name, keep in eligible.items():
        if not keep.any():
            raise ValueError(f"No training support for {name}: {scheme}/{model}")
        if name != "lookup_all" and (keep & hidden_rows).any():
            raise ValueError(f"Hidden target leakage in {name}: {model}")
        lookups[name] = float(data.loc[keep, "delta"].mean())
        counts[name] = int(keep.sum())

    card = own.groupby("benchmark")[["acc_before", "acc_after", "delta"]].mean()
    records = []
    for benchmark, (_, column) in zip(hidden, cells):
        actual = float(matrix.iloc[row, column])
        base = float(matrix.iloc[base_row, column])
        card_base = float(card.loc[benchmark, "acc_before"])
        prediction = float(predicted[row, column])
        np.testing.assert_allclose(actual, card.loc[benchmark, "acc_after"],
                                   rtol=0, atol=1e-10)
        records.append({
            "seed": seed, "model": model, "base_model": base_model,
            "family": str(own.family.iloc[0]),
            "scheme": scheme, "benchmark": benchmark, "revealed": revealed,
            "actual_score": actual, "predicted_score": prediction,
            "base_matrix_score": base, "base_card_score": card_base,
            "delta_matrix": actual - base, "delta_card": actual - card_base,
            "predicted_delta_matrix": prediction - base,
            "predicted_delta_card": prediction - card_base,
            "zero_delta": 0.0, **lookups, "lookup_training_counts": counts,
        })
    score_records(records, "actual_score", ["predicted_score"])
    write_json_atomic(path, {
        "config": config, "specification": specification, "records": records,
    })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["sanity", "transfer"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    commit = subprocess.check_output(
        ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
            ["git", "-C", REPO_ROOT, "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run from a clean committed checkout")
    config = {
        "stage": args.stage, "code_commit": commit,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "predictor": "logit_bias_als", "rank": 2, "lambda": 0.1,
        "ensemble_seeds": list(range(42, 52)), "iterations": 40,
    }
    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    os.makedirs(output_root, exist_ok=True)

    if args.stage == "sanity":
        folds_path = os.path.join(FOLDS_DIR, "folds_s10_f3_bs42_ms1.json")
        if not os.path.exists(folds_path):
            raise FileNotFoundError(f"Existing canonical folds required: {folds_path}")
        folds = load_folds()
        if (M_FULL.shape != (84, 133) or len(folds) != 30
                or int(np.isfinite(M_FULL).sum()) != 2604):
            raise ValueError("Sanity requires the original 84x133 paper snapshot")
        with open(REFERENCE_PREDICTIONS, "rb") as handle:
            reference_hash = hashlib.sha256(handle.read()).hexdigest()
        with np.load(REFERENCE_PREDICTIONS, allow_pickle=False) as reference:
            metadata = json.loads(str(reference["metadata_json"]))
            expected = {"hp": {"lam": 0.1, "rank": 2}, "method": "Bias ALS",
                        "transform": "logit", "n_folds": 3, "n_seeds": 10,
                        "base_seed": 42, "matrix_shape": [84, 133]}
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError("Reference predictor configuration mismatch")
            expected_cells = np.asarray([
                (fold, i, j) for fold, (_, cells) in enumerate(folds) for i, j in cells
            ])
            cached_cells = np.column_stack([
                reference["fold_id"], reference["test_i"], reference["test_j"]])
            np.testing.assert_array_equal(cached_cells, expected_cells)
            np.testing.assert_array_equal(
                reference["actual"], M_FULL[cached_cells[:, 1], cached_cells[:, 2]])
            cached_matrices = reference["M_pred_by_fold"]
            np.testing.assert_array_equal(
                reference["predicted"], cached_matrices[
                    cached_cells[:, 0], cached_cells[:, 1], cached_cells[:, 2]])
            if cached_matrices.shape != (30, 84, 133):
                raise ValueError("Unexpected cached prediction matrix shape")
        with open(folds_path, "rb") as handle:
            folds_hash = hashlib.sha256(handle.read()).hexdigest()
        config.update({
            "matrix_sha256": matrix_identity_sha256(M_FULL),
            "metric_sha256": benchmark_metric_identity_sha256(),
            "folds_sha256": folds_hash, "seeds": list(range(42, 52)),
            "n_folds": 3, "n_test_predictions": sum(len(cells) for _, cells in folds),
            "reference_sha256": reference_hash,
            "validation": "all reference targets/masks; current predictor on fold 0",
        })
        jobs = [(fold, matrix, cells, cached_matrices[fold])
                for fold, (matrix, cells) in enumerate(folds)]
        input_report = {"model_count": 84, "benchmark_ids": BENCH_IDS,
                        "benchmark_metrics": BENCH_METRICS,
                        "observed_cells": int(np.isfinite(M_FULL).sum())}
    else:
        data_path = os.path.join(output_root, "dataset.csv")
        if not os.path.exists(data_path):
            with urlopen(DATA_URL, timeout=60) as response:
                content = response.read()
            with open(data_path + ".tmp", "wb") as handle:
                handle.write(content)
            os.replace(data_path + ".tmp", data_path)
        with open(data_path, "rb") as handle:
            content = handle.read()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != DATA_BLOB:
            raise ValueError(f"Dataset does not match pinned source: {blob}")
        raw = pd.read_csv(data_path)
        data = raw.loc[raw.acc_before >= 20].reset_index(drop=True)
        np.testing.assert_allclose(data.delta, data.acc_after - data.acc_before,
                                   rtol=0, atol=1e-8)
        base = data.groupby(["base_model", "benchmark"]).acc_before.mean().unstack()
        base.index = "BASE::" + base.index
        quant = data.groupby(["model", "benchmark"]).acc_after.mean().unstack()
        quant.index = "QUANT::" + quant.index
        matrix = pd.concat([base, quant]).sort_index().sort_index(axis=1)
        if matrix.shape != (140, 16):
            raise ValueError(f"Unexpected source matrix: {matrix.shape}")
        metadata = data.groupby("model").base_model.first()
        jobs = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            for model, base_model in metadata.items():
                qrow = matrix.loc["QUANT::" + model]
                brow = matrix.loc["BASE::" + base_model]
                observed = list(matrix.columns[qrow.notna() & brow.notna()])
                if len(observed) < 5:
                    continue
                permutation = list(rng.permutation(observed))
                revealed = permutation[:3]
                hidden = [name for name in observed if name not in revealed]
                jobs.append((seed, model, revealed, hidden))
        config.update({
            "source_commit": SOURCE_COMMIT, "data_url": DATA_URL,
            "dataset_sha256": hashlib.sha256(content).hexdigest(),
            "dataset_git_blob": blob, "min_acc_before": 20,
            "seeds": list(range(5)), "known_scores": 3,
            "minimum_paired_scores": 5, "drop_siblings": False,
            "matrix_sha256": matrix_identity_sha256(matrix.to_numpy(float)),
            "lookup_target": "original card delta",
            "benchpress_row_exclusions": "none; siblings and base row retained",
            "lookup_exclusions": {
                "lookup_all": "none; leaky reproduction control",
                "lookup_hidden_excluded": "hidden target cells only; matched information",
                "lookup_checkpoint_excluded": "lookup only: target checkpoint",
                "lookup_base_excluded": "lookup only: target base-model group",
            },
        })
        spread = data.groupby(["base_model", "benchmark"]).acc_before.agg(
            ["min", "max", "count"])
        spread["range"] = spread["max"] - spread["min"]
        input_report = {
            "source_rows": len(raw), "retained_rows": len(data),
            "excluded_near_random_rows": len(raw) - len(data),
            "matrix_shape": list(matrix.shape),
            "observed_cells": int(matrix.notna().to_numpy().sum()),
            "families": int(data.family.nunique()),
            "n_target_units": len(jobs),
            "n_disagreeing_base_groups": int((spread["range"] > 1e-8).sum()),
            "max_base_score_range": float(spread["range"].max()),
            "base_score_disagreements": spread.loc[spread["range"] > 1e-8]
                .reset_index().sort_values("range", ascending=False).to_dict("records"),
        }

    output_dir = os.path.join(output_root, args.stage, hp_short_hash(config, n=16))
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {"config": config, "inputs": input_report, "expected_units": len(jobs)}
    if os.path.exists(manifest_path) and load_json(manifest_path) != manifest:
        raise ValueError(f"Conflicting manifest: {manifest_path}")
    write_json_atomic(manifest_path, manifest, indent=2)
    print("[CONFIG]", canonical_hp_json(config), flush=True)
    print("[INPUTS]", canonical_hp_json(input_report), flush=True)
    worker = (partial(run_sanity_fold, output_dir=output_dir, config=config)
              if args.stage == "sanity" else
              partial(run_transfer_target, data=data, matrix=matrix,
                      output_dir=output_dir, config=config))
    records = worker(jobs[0])
    print(f"[PROGRESS] 1/{len(jobs)} units; first unit validated", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, rows in enumerate(executor.map(worker, jobs[1:]), start=2):
            records.extend(rows)
            if index % 20 == 0 or index == len(jobs):
                print(f"[PROGRESS] {index}/{len(jobs)} units", flush=True)
    if args.stage == "sanity":
        summary = {"reference_metrics": compute_prediction_error(
            np.array([row["actual"] for row in records]),
            np.array([row["predicted"] for row in records]),
            groups=np.array([row["fold"] for row in records]),
            aggregation="per_group_median"),
            "fresh_fold": score_records([row for row in records if row["fold"] == 0],
                                        "actual", ["current_predicted", "predicted"]),
            "fresh_fold_max_prediction_difference": float(max(
                abs(row["current_predicted"] - row["predicted"])
                for row in records if row["fold"] == 0)),
        }
    else:
        keys = ["zero_delta", "lookup_all", "lookup_hidden_excluded",
                "lookup_checkpoint_excluded", "lookup_base_excluded"]
        summary = {
            truth: score_records(records, f"delta_{truth}",
                                 [f"predicted_delta_{truth}", *keys])
            for truth in ["matrix", "card"]
        }
        summary["per_seed"] = {
            truth: {
                str(seed): score_records(
                    [row for row in records if row["seed"] == seed],
                    f"delta_{truth}", [f"predicted_delta_{truth}", *keys])
                for seed in range(5)
            } for truth in ["matrix", "card"]
        }
        summary["per_benchmark"] = {
            benchmark: score_records(
                [row for row in records if row["benchmark"] == benchmark],
                "delta_card", ["predicted_delta_card", "lookup_hidden_excluded"])
            for benchmark in sorted({row["benchmark"] for row in records})
        }
        summary["unique_target_cells"] = len({
            (row["model"], row["benchmark"]) for row in records})
    write_json_atomic(os.path.join(output_dir, "records.json"), records)
    write_json_atomic(os.path.join(output_dir, "summary.json"), summary, indent=2)
    print("[SUMMARY]", canonical_hp_json(summary), flush=True)
    print("[OUTPUT]", output_dir, flush=True)


if __name__ == "__main__":
    main()
