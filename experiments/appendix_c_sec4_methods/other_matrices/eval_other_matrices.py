#!/usr/bin/env python
"""Score BenchPress on externally curated score matrices (HELM, EEE).

Thin wrapper over the shared harness: it reuses `holdout_per_model` (now matrix
parameterized) for the paper's 10 seeds x 3 folds per-model holdout,
`predict_benchpress_scores` for the predictor, and `compute_prediction_error`
with per-fold-median aggregation, exactly like the main method comparison. Only
the matrix prep is experiment specific: load a curated `scores.csv`, keep the
percentage-scale columns (so a point-scale MedAE is meaningful and the logit
transform applies), and apply the paper's (>=15, >=8) observation filter.

Backs `tab:other_matrices` in `app:other_matrices`.

Usage:
    python eval_other_matrices.py --matrix helm=~/helm_matrix --matrix eee=~/eee_matrix
"""
import argparse
import json
import os

import numpy as np

from benchpress.evaluation_harness import holdout_per_model, compute_prediction_error
from benchpress.methods.predictors import predict_benchpress_scores
from benchpress.data.score_matrix import ScoreMatrix

N_SEEDS, N_FOLDS, BASE_SEED, MIN_SCORES = 10, 3, 42, 1
MIN_BENCH_PER_MODEL, MIN_MODELS_PER_BENCH = 15, 8
PCT_TYPES = {"pct", "percent", "percentage"}


def load_pct_matrix(path):
    """Load a curated matrix and keep only its percentage-scale benchmark columns."""
    if os.path.isdir(path):
        path = os.path.join(path, "scores.csv")
    sm = ScoreMatrix.from_file(path)
    M = np.asarray(sm.values, dtype=float)
    benches = list(sm.benchmark_ids)
    pct = np.array([str(sm.metric.get(b, {}).get("type", "")).lower() in PCT_TYPES
                    for b in benches])
    return M[:, pct]


def threshold_filter(M, min_bench=MIN_BENCH_PER_MODEL, min_models=MIN_MODELS_PER_BENCH):
    """Iterate the paper's (min_bench per model, min_models per benchmark) filter."""
    while True:
        obs = np.isfinite(M)
        row_ok = obs.sum(axis=1) >= min_bench
        col_ok = obs.sum(axis=0) >= min_models
        if row_ok.all() and col_ok.all():
            return M
        M = M[row_ok][:, col_ok]
        if M.size == 0:
            return M


def benchmark_median(M_train):
    global_med = float(np.nanmedian(M_train))
    with np.errstate(invalid="ignore"):
        col_med = np.nanmedian(M_train, axis=0)
    col_med = np.where(np.isnan(col_med), global_med, col_med)
    return np.tile(col_med, (M_train.shape[0], 1))


def global_median(M_train):
    return np.full_like(M_train, float(np.nanmedian(M_train)), dtype=np.float64)


PREDICTORS = {
    "benchpress": predict_benchpress_scores,
    "benchmark_median": benchmark_median,
    "global_median": global_median,
}


def evaluate(M):
    folds = []
    for s in range(N_SEEDS):
        folds.extend(holdout_per_model(min_scores=MIN_SCORES, n_folds=N_FOLDS,
                                       seed=BASE_SEED + s, M=M))
    preds = {name: {} for name in PREDICTORS}
    fold_id, test_set = [], []
    for fid, (M_train, cells) in enumerate(folds):
        for name, fn in PREDICTORS.items():
            preds[name][fid] = np.asarray(fn(M_train), dtype=np.float64)
        fold_id.extend([fid] * len(cells))
        test_set.extend(cells)
    scored = {}
    for name, per_fold in preds.items():
        res = compute_prediction_error(M, per_fold, test_set=test_set,
                                       groups=fold_id, aggregation="per_group_median")
        scored[name] = {"medae": res["medae_median"], "medape": res["medape_median"]}
    n_models, n_bench = M.shape
    observed = int(np.isfinite(M).sum())
    return {
        "matrix_shape": [int(n_models), int(n_bench)],
        "observed_cells": observed,
        "fill_rate": observed / (n_models * n_bench),
        "n_held_out_cells": len(test_set),
        "results": scored,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results.json"))
    args = parser.parse_args()

    out = {"matrices": {}}
    for spec in args.matrix:
        label, path = spec.split("=", 1)
        M = threshold_filter(load_pct_matrix(os.path.expanduser(path)))
        print(f"evaluating {label} matrix {M.shape} "
              f"({int(np.isfinite(M).sum())} cells)", flush=True)
        out["matrices"][label] = evaluate(M)

    with open(os.path.expanduser(args.out), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["matrices"], indent=2))


if __name__ == "__main__":
    main()
