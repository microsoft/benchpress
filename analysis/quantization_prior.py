"""Test prior-centered residual prediction on saved quantization holdouts.

Run on GCR, with its benchpress environment, after committing and pushing:

    cd ~/projects/BenchPress
    python -m analysis.quantization_prior --phase smoke \
        --data /tmp/benchpress-quantization-transfer/dataset.csv \
        --reference-dir /tmp/benchpress-quantization-transfer/transfer/5b3945cc9e4255ce \
        --output-root /tmp/benchpress-quantization-prior

Then use --phase evaluate --workers 8. Smoke computes the first full target,
including nested validation, and is reused by the full run.

Targets and revealed cells come from the previous 5-seed transfer experiment.
Only card-paired deltas enter the matrix; no averaged base row is constructed.
The prior is each scheme's training-only mean and predictive scale
sqrt(sample_variance * (1 + 1/n)). This plug-in scale includes uncertainty in
the estimated mean under an iid working model; it is not a calibrated
posterior interval or a correction for dependence between model cards.

Fit rank-2 Bias ALS to (delta - prior_mean) / prior_scale without logit or
column normalization. Select a correction multiplier in {0, .25, .5, .75, 1}
by inner validation MAE. Each outer target uses three disjoint sets of ten
other eligible checkpoints; jointly hide all but three observations on each
inner checkpoint. All outer-test cells stay missing throughout inner fitting.
Refit the prior and residual model with the inner cells restored, never the
outer targets. Exact ties choose the smaller correction. No test-based tuning.
At weight one, ALS biases can substantially displace the prior mean; the
explicit mean anchoring comes from the correction multiplier below one.
This is an exploratory follow-up designed after inspecting the previous test
results, not a fresh confirmation on untouched data.

Save every outer and inner prediction, prior support, selected multiplier,
candidate loss, and input/code/config identity. Old absolute-score predictions
are read from the reference artifact rather than recomputed.
"""

import argparse
import hashlib
import os
import platform
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from benchpress.evaluation_harness import mask_cells, matrix_identity_sha256
from benchpress.io_utils import load_json, write_json_atomic
from benchpress.methods.completers import complete_bias_als
from benchpress.methods.predictors import predict_prior_residual_scores
from benchpress.shard_utils import hp_short_hash
from benchpress.stats import cluster_bootstrap_mean
from analysis.quantization_transfer import DATA_BLOB, REPO_ROOT, SOURCE_COMMIT, score_records


CORRECTION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = (
    "original_benchpress", "zero_delta", "prior_only", "delta_als",
    "delta_als_unnormalized",
    "prior_residual_unshrunk", "prior_residual_selected",
)


def estimate_scheme_prior(observed_deltas, schemes):
    """Return per-cell prior means/scales and per-scheme observed-data moments."""
    if len(schemes) != observed_deltas.shape[0]:
        raise ValueError("One scheme is required per matrix row")
    means = np.empty_like(observed_deltas)
    scales = np.empty_like(observed_deltas)
    moments = {}
    for scheme in np.unique(schemes):
        selected = schemes == scheme
        values = observed_deltas[selected]
        values = values[np.isfinite(values)]
        if len(values) < 2:
            raise ValueError(f"Insufficient prior support: {scheme}")
        mean = float(values.mean())
        variance = float(values.var(ddof=1))
        if variance <= 0 or not np.isfinite(variance):
            raise ValueError(f"Nonpositive prior variance: {scheme}")
        scale = float(np.sqrt(variance * (1 + 1 / len(values))))
        means[selected] = mean
        scales[selected] = scale
        moments[str(scheme)] = {"n": len(values), "mean": mean,
                                "sample_variance": variance, "scale": scale}
    return means, scales, moments


def select_correction_weight(observed_deltas, schemes, target_row, seed):
    """Choose residual shrinkage using only an already outer-masked matrix."""
    candidates = np.flatnonzero(np.isfinite(observed_deltas).sum(axis=1) >= 5)
    candidates = candidates[candidates != target_row]
    if len(candidates) < 30:
        raise ValueError("Nested validation requires 30 other eligible checkpoints")
    rng = np.random.default_rng(np.random.SeedSequence([1729, seed, target_row]))
    validation_rows = rng.permutation(candidates)[:30].reshape(3, 10)
    records, folds = [], []
    for fold, rows in enumerate(validation_rows):
        hidden, revealed = [], {}
        for row in rows:
            observed = np.flatnonzero(np.isfinite(observed_deltas[row]))
            known = rng.permutation(observed)[:3]
            revealed[str(int(row))] = [int(column) for column in known]
            hidden.extend((int(row), int(column)) for column in observed
                          if column not in known)
        inner_train = mask_cells(hidden, base_matrix=observed_deltas)
        if np.isfinite(inner_train[np.isnan(observed_deltas)]).any():
            raise ValueError("Outer hidden cells leaked into inner training")
        mean, scale, moments = estimate_scheme_prior(inner_train, schemes)
        predicted = predict_prior_residual_scores(inner_train, mean, scale)
        for row, column in hidden:
            records.append({
                "fold": fold, "row": row, "column": column,
                "actual": float(observed_deltas[row, column]),
                "prior": float(mean[row, column]),
                "unshrunk": float(predicted[row, column]),
            })
        folds.append({"fold": fold, "revealed_columns_by_row": revealed,
                      "training_sha256": matrix_identity_sha256(inner_train),
                      "prior_moments": moments})
    actual = np.array([record["actual"] for record in records])
    prior = np.array([record["prior"] for record in records])
    correction = np.array([record["unshrunk"] for record in records]) - prior
    losses = [float(mean_absolute_error(actual, prior + weight * correction))
              for weight in CORRECTION_WEIGHTS]
    selected = CORRECTION_WEIGHTS[int(np.argmin(losses))]
    return selected, {"folds": folds, "records": records,
                      "weights": list(CORRECTION_WEIGHTS), "mae": losses}


def run_target(job, matrix, schemes, output_dir, config):
    """Persist one matched outer target, its nested selection, and ablations."""
    row, reference_rows, benchmark_ids = job
    reference = reference_rows[0]
    specification = {
        "seed": reference["seed"], "model": reference["model"],
        "revealed": reference["revealed"],
        "hidden": [record["benchmark"] for record in reference_rows],
    }
    path = os.path.join(output_dir, f"target_{hp_short_hash(specification)}.json")
    if os.path.exists(path):
        payload = load_json(path)
        if payload["config"] != config or payload["specification"] != specification:
            raise ValueError(f"Stale prior artifact: {path}")
        if [record["benchmark"] for record in payload["records"]] != specification["hidden"]:
            raise ValueError(f"Incomplete prior artifact: {path}")
        score_records(payload["records"], "actual_delta", METHODS)
        return payload
    started = time.monotonic()
    cells = [(row, benchmark_ids.index(name)) for name in specification["hidden"]]
    train = mask_cells(cells, base_matrix=matrix)
    visible = [benchmark_ids[column] for column in np.flatnonzero(np.isfinite(train[row]))]
    if set(visible) != set(specification["revealed"]) or len(visible) != 3:
        raise ValueError("Reference reveal mask does not match the delta matrix")
    with threadpool_limits(limits=1):
        weight, validation = select_correction_weight(
            train, schemes, row, specification["seed"])
        mean, scale, moments = estimate_scheme_prior(train, schemes)
        unshrunk = predict_prior_residual_scores(train, mean, scale)
        direct = complete_bias_als(train, rank=2, lam=0.1, normalize=True)
        unnormalized = complete_bias_als(train, rank=2, lam=0.1, normalize=False)
    records = []
    for source, (_, column) in zip(reference_rows, cells):
        actual = float(matrix[row, column])
        np.testing.assert_allclose(actual, source["delta_card"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            mean[row, column], source["lookup_hidden_excluded"], rtol=0, atol=1e-10)
        selected = float(mean[row, column] + weight * (unshrunk[row, column] - mean[row, column]))
        records.append({
            "seed": specification["seed"], "model": specification["model"],
            "base_model": source["base_model"], "family": source["family"],
            "scheme": source["scheme"], "benchmark": source["benchmark"],
            "revealed": specification["revealed"], "actual_delta": actual,
            "base_card_score": source["base_card_score"],
            "actual_score": source["actual_score"],
            "selected_score": source["base_card_score"] + selected,
            "original_benchpress": source["predicted_delta_card"],
            "zero_delta": 0.0, "prior_only": float(mean[row, column]),
            "delta_als": float(direct[row, column]),
            "delta_als_unnormalized": float(unnormalized[row, column]),
            "prior_residual_unshrunk": float(unshrunk[row, column]),
            "prior_residual_selected": selected, "correction_weight": weight,
            "prior_scale": float(scale[row, column]),
            "prior_n": moments[source["scheme"]]["n"],
        })
    score_records(records, "actual_delta", METHODS)
    for record in records:
        record["out_of_domain_methods"] = [
            method for method in METHODS
            if not 0 <= record["base_card_score"] + record[method] <= 100]
    payload = {
        "config": config, "specification": specification, "records": records,
        "selected_weight": weight, "validation": validation,
        "outer_training_sha256": matrix_identity_sha256(train),
        "outer_prior_moments": moments, "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "evaluate"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if subprocess.check_output(
            ["git", "-C", REPO_ROOT, "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run from a clean committed canonical checkout")
    commit = subprocess.check_output(
        ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    with open(os.path.expanduser(args.data), "rb") as handle:
        content = handle.read()
    if hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() != DATA_BLOB:
        raise ValueError("Dataset does not match the pinned public corpus")
    reference_dir = os.path.abspath(os.path.expanduser(args.reference_dir))
    reference_manifest = load_json(os.path.join(reference_dir, "manifest.json"))
    with open(os.path.join(reference_dir, "records.json"), "rb") as handle:
        reference_hash = hashlib.sha256(handle.read()).hexdigest()
    expected = {"source_commit": SOURCE_COMMIT, "min_acc_before": 20,
                "seeds": list(range(5)), "known_scores": 3, "drop_siblings": False,
                "dataset_sha256": hashlib.sha256(content).hexdigest()}
    if any(reference_manifest["config"].get(key) != value for key, value in expected.items()):
        raise ValueError("Reference protocol or input identity mismatch")
    data = pd.read_csv(os.path.expanduser(args.data))
    data = data.loc[data.acc_before >= 20]
    if data.duplicated(["model", "benchmark"]).any() or len(data) != 817:
        raise ValueError("Expected 817 unique card-paired observations")
    np.testing.assert_allclose(data.delta, data.acc_after - data.acc_before,
                               rtol=0, atol=1e-10)
    frame = data.pivot(index="model", columns="benchmark", values="delta")
    model_ids, benchmark_ids = list(frame.index), list(frame.columns)
    matrix = frame.to_numpy(float)
    metadata = data.groupby("model").scheme.agg(["first", "nunique"]).reindex(model_ids)
    if (metadata["nunique"] != 1).any():
        raise ValueError("Ambiguous scheme for a checkpoint")
    schemes = metadata["first"].to_numpy()
    references = load_json(os.path.join(reference_dir, "records.json"))
    grouped = {}
    for record in references:
        grouped.setdefault((record["seed"], record["model"]), []).append(record)
    if len(references) != 2555 or len(grouped) != 505:
        raise ValueError("Expected the complete 5-seed reference holdouts")
    jobs = [(model_ids.index(model), rows, benchmark_ids)
            for (seed, model), rows in sorted(grouped.items())]
    config = {
        "code_commit": commit, "source_commit": SOURCE_COMMIT,
        "dataset_sha256": hashlib.sha256(content).hexdigest(),
        "reference_records_sha256": reference_hash,
        "delta_matrix_sha256": matrix_identity_sha256(matrix),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "prior_mean": "scheme mean of outer/inner visible card deltas",
        "prior_scale": "sqrt(unbiased variance * (1 + 1/n))",
        "residual_transform": "identity; prior-scale only; no column normalization",
        "rank": 2, "lambda": 0.1, "iterations": 40,
        "ensemble_seeds": list(range(42, 52)), "outer_seeds": list(range(5)),
        "correction_weights": list(CORRECTION_WEIGHTS),
        "inner_seed": 1729, "inner_folds": 3, "inner_checkpoints_per_fold": 10,
        "inner_selection": "pooled MAE; ties prefer smaller weight",
        "known_scores": 3, "min_acc_before": 20, "drop_siblings": False,
        "direct_delta_ablation": "identity; column normalization; rank2 lambda0.1",
        "unnormalized_delta_ablation": "identity; no normalization; rank2 lambda0.1",
        "bootstrap": {"cluster": "base_model", "resamples": 2000, "seed": 42,
                      "confidence": 0.95, "estimand": "selected MAE minus prior MAE"},
        "study_scope": "exploratory follow-up; not untouched test data",
    }
    output_dir = os.path.join(os.path.abspath(os.path.expanduser(args.output_root)),
                              hp_short_hash(config, n=16))
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"config": config, "model_ids": model_ids,
                "benchmark_ids": benchmark_ids, "expected_units": len(jobs),
                "expected_predictions": len(references), "methods": list(METHODS),
                "reference_config": reference_manifest["config"]}
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path) and load_json(manifest_path) != manifest:
        raise ValueError("Conflicting output manifest")
    write_json_atomic(manifest_path, manifest, indent=2)
    print("[CONFIG]", config, flush=True)
    worker = partial(run_target, matrix=matrix, schemes=schemes,
                     output_dir=output_dir, config=config)
    first = worker(jobs[0])
    print("[FIRST_UNIT]", first["elapsed_seconds"], "seconds",
          score_records(first["records"], "actual_delta", METHODS), flush=True)
    if args.phase == "smoke":
        print("[OUTPUT]", output_dir, flush=True)
        return
    records = list(first["records"])
    weights = Counter([first["selected_weight"]])
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, payload in enumerate(executor.map(worker, jobs[1:]), start=2):
            records.extend(payload["records"])
            weights[payload["selected_weight"]] += 1
            if completed % 20 == 0 or completed == len(jobs):
                print(f"[PROGRESS] {completed}/{len(jobs)}", flush=True)
    if len({(r["seed"], r["model"], r["benchmark"]) for r in records}) != 2555:
        raise ValueError("Final predictions contain missing or duplicate targets")
    summary = {
        "overall": score_records(records, "actual_delta", METHODS),
        "per_seed": {
            str(seed): score_records([r for r in records if r["seed"] == seed],
                                    "actual_delta", METHODS) for seed in range(5)},
        "per_benchmark": {
            name: score_records([r for r in records if r["benchmark"] == name],
                                "actual_delta", METHODS)
            for name in sorted({r["benchmark"] for r in records})},
        "per_scheme": {
            name: score_records([r for r in records if r["scheme"] == name],
                                "actual_delta", METHODS)
            for name in sorted({r["scheme"] for r in records})},
        "selected_weight_counts": {str(weight): weights[weight] for weight in CORRECTION_WEIGHTS},
        "unique_target_cells": len({(r["model"], r["benchmark"]) for r in records}),
        "out_of_domain_counts": {
            method: sum(method in r["out_of_domain_methods"] for r in records)
            for method in METHODS},
    }
    actual = np.array([r["actual_delta"] for r in records])
    prior = np.array([r["prior_only"] for r in records])
    residual = np.array([r["prior_residual_unshrunk"] for r in records]) - prior
    selected = np.array([r["prior_residual_selected"] for r in records])
    summary["fixed_weight_outer_mae_diagnostic_only"] = {
        str(weight): float(mean_absolute_error(actual, prior + weight * residual))
        for weight in CORRECTION_WEIGHTS}
    summary["selected_minus_prior_cluster_bootstrap"] = cluster_bootstrap_mean(
        np.abs(actual - selected) - np.abs(actual - prior),
        [r["base_model"] for r in records], n_resamples=2000, seed=42, confidence=0.95)
    write_json_atomic(os.path.join(output_dir, "records.json"), records)
    write_json_atomic(os.path.join(output_dir, "summary.json"), summary, indent=2)
    print("[SUMMARY]", summary["overall"], flush=True)
    print("[WEIGHTS]", summary["selected_weight_counts"], flush=True)
    print("[OUTPUT]", output_dir, flush=True)


if __name__ == "__main__":
    main()
