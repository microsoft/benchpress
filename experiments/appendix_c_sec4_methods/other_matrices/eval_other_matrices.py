#!/usr/bin/env python
"""Evaluate BenchPress and median baselines on externally curated score matrices.

For each curated matrix (a `scores.csv` from `benchpress.data.<source>.curate_matrix`),
we apply the paper's construction filter (>=15 observed benchmarks per model,
>=8 observed models per benchmark, iterated to a fixed point) and then score the
canonical rank-2 logit Bias ALS predictor (BenchPress) against a per-benchmark
median and a global median baseline, under the paper's 10 seeds x 3 folds
per-model holdout with per-fold-median aggregation (identical protocol to the
main method comparison). Backs `tab:other_matrices` in `app:other_matrices`.

Usage:
    python eval_other_matrices.py --matrix helm=~/helm_matrix --matrix eee=~/eee_matrix
"""
import argparse
import json
import os

import numpy as np

from benchpress.evaluation_harness import compute_prediction_error, make_score_predictor
from benchpress.methods.completers import complete_bias_als
from benchpress.data.score_matrix import ScoreMatrix

TRANSFORM = "logit"
HP = {"rank": 2, "lam": 0.1}
N_SEEDS, N_FOLDS, BASE_SEED, MIN_SCORES = 10, 3, 42, 1
MIN_BENCH_PER_MODEL, MIN_MODELS_PER_BENCH = 15, 8


def benchpress_predictor():
    method_fn = lambda M, **kw: complete_bias_als(M, normalize=False, **kw)
    return make_score_predictor(method_fn, TRANSFORM, **HP)


def benchmark_median(M_train):
    """Predict every cell of column j by the median of column j's training scores."""
    global_med = float(np.nanmedian(M_train))
    with np.errstate(invalid="ignore"):
        col_med = np.nanmedian(M_train, axis=0)
    empty = np.isnan(col_med)
    col_med = np.where(empty, global_med, col_med)
    return np.tile(col_med, (M_train.shape[0], 1)), int(empty.sum())


def global_median(M_train):
    """Predict every cell by one scalar, the median of all training scores."""
    return np.full_like(M_train, float(np.nanmedian(M_train)), dtype=np.float64), 0


BASELINES = {"benchmark_median": benchmark_median, "global_median": global_median}


def threshold_filter(M, models, benchmarks,
                     min_bench=MIN_BENCH_PER_MODEL, min_models=MIN_MODELS_PER_BENCH):
    """Iterate the paper's (min_bench, min_models) observation filter to a fixed point."""
    models = list(models)
    benchmarks = list(benchmarks)
    while True:
        obs = np.isfinite(M)
        row_ok = obs.sum(axis=1) >= min_bench
        col_ok = obs.sum(axis=0) >= min_models
        if row_ok.all() and col_ok.all():
            break
        M = M[row_ok][:, col_ok]
        models = [m for m, ok in zip(models, row_ok) if ok]
        benchmarks = [b for b, ok in zip(benchmarks, col_ok) if ok]
        if M.size == 0:
            break
    return M, models, benchmarks


def per_model_folds(M_full, n_seeds, n_folds, base_seed, min_scores):
    """Per-model holdout folds for an arbitrary matrix, matching holdout_per_model."""
    observed = ~np.isnan(M_full)
    n_models = M_full.shape[0]
    folds = []
    for seed_idx in range(n_seeds):
        rng = np.random.RandomState(base_seed + seed_idx)
        assignments = []
        for i in range(n_models):
            obs_j = list(np.where(observed[i])[0])
            if len(obs_j) >= min_scores:
                rng.shuffle(obs_j)
                assignments.append(obs_j)
            else:
                assignments.append([])
        for k in range(n_folds):
            M_train = M_full.copy()
            test_set = []
            for i in range(n_models):
                obs_j = assignments[i]
                if not obs_j:
                    continue
                fold_size = max(1, len(obs_j) // n_folds)
                start = k * fold_size
                end = start + fold_size if k < n_folds - 1 else len(obs_j)
                if start >= len(obs_j):
                    continue
                for j in obs_j[start:end]:
                    M_train[i, j] = np.nan
                    test_set.append((i, j))
            folds.append((M_train, test_set))
    return folds


def evaluate(M_full, folds, label):
    """Score BenchPress and both baselines on identical held-out cells."""
    predict_fn = benchpress_predictor()
    names = ["benchpress"] + list(BASELINES)
    preds = {name: {} for name in names}
    fold_id, rows, cols = [], [], []
    n_empty_columns = 0
    for fid, (M_train, test_set) in enumerate(folds):
        preds["benchpress"][fid] = predict_fn(M_train).astype(np.float64, copy=False)
        for name, fn in BASELINES.items():
            M_pred, n_empty = fn(M_train)
            preds[name][fid] = M_pred.astype(np.float64, copy=False)
            if name == "benchmark_median":
                n_empty_columns += n_empty
        for i, j in test_set:
            fold_id.append(fid)
            rows.append(i)
            cols.append(j)
        if (fid + 1) % 10 == 0 or fid + 1 == len(folds):
            print(f"  [{label}] {fid + 1}/{len(folds)} folds", flush=True)
    fold_id = np.asarray(fold_id, dtype=int)
    test_set = list(zip(rows, cols))
    scored = {}
    for name, per_fold in preds.items():
        res = compute_prediction_error(M_full, per_fold, test_set=test_set,
                                       groups=fold_id.tolist(),
                                       aggregation="per_group_median")
        scored[name] = {
            "medae": res["medae_median"],
            "medape": res["medape_median"],
            "coverage": sum(g["n"] for g in res["per_group"].values()) / len(test_set),
        }
    bp = scored["benchpress"]
    for row in scored.values():
        row["medae_ratio_to_benchpress"] = row["medae"] / bp["medae"]
        row["medape_ratio_to_benchpress"] = row["medape"] / bp["medape"]
    n_models, n_bench = M_full.shape
    observed = int(np.isfinite(M_full).sum())
    return {
        "matrix_shape": [int(n_models), int(n_bench)],
        "observed_cells": observed,
        "fill_rate": observed / (n_models * n_bench),
        "n_folds": len(folds),
        "n_held_out_cells": len(test_set),
        "empty_training_column_fallbacks": n_empty_columns,
        "results": scored,
    }


def load_matrix(path):
    if os.path.isdir(path):
        path = os.path.join(path, "scores.csv")
    sm = ScoreMatrix.from_file(path)
    return np.asarray(sm.values, dtype=float), list(sm.model_ids), list(sm.benchmark_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="append", default=[], metavar="LABEL=PATH",
                        help="repeatable, e.g. --matrix helm=~/helm_matrix")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results.json"))
    args = parser.parse_args()

    out = {
        "config": {"transform": TRANSFORM, "hp": HP, "n_seeds": N_SEEDS,
                   "n_folds": N_FOLDS, "base_seed": BASE_SEED, "min_scores": MIN_SCORES,
                   "min_bench_per_model": MIN_BENCH_PER_MODEL,
                   "min_models_per_bench": MIN_MODELS_PER_BENCH},
        "matrices": {},
    }
    for spec in args.matrix:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        M, models, benches = load_matrix(path)
        M, models, benches = threshold_filter(M, models, benches)
        print(f"evaluating {label} matrix {M.shape} "
              f"({int(np.isfinite(M).sum())} cells) from {path}", flush=True)
        folds = per_model_folds(M, N_SEEDS, N_FOLDS, BASE_SEED, MIN_SCORES)
        summary = evaluate(M, folds, label)
        summary["source_path"] = path
        out["matrices"][label] = summary

    with open(os.path.expanduser(args.out), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["matrices"], indent=2))


if __name__ == "__main__":
    main()
