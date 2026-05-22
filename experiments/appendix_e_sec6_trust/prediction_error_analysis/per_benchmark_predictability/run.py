#!/usr/bin/env python3
"""
Per-benchmark predictability via half-per-model holdout.

For each of the 49 benchmarks, compute how well BenchPress can predict its
scores using the standard half-per-model holdout: for every model, hide 50%
of observed scores (randomly), predict with BenchPress, then aggregate
test-cell errors *per benchmark column*.

This answers: "Which benchmarks are most/least predictable from the others?"

Multiple seeds for stability.  Output: results.json with per-benchmark metrics.
"""

import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH,
    BENCH_IDS, BENCH_NAMES, BENCH_CATS,
    compute_prediction_error,
    holdout_half_per_model,
)
from benchpress.all_methods import predict_benchpress_scores
from benchpress.io_utils import write_json

N_SEEDS = 10


def run():
    # Per-seed raw predictions — THE bottleneck output
    # Each seed: list of (model_idx, bench_idx, actual, predicted)
    raw_predictions = {seed: [] for seed in range(N_SEEDS)}

    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        M_train, test_cells = holdout_half_per_model(rng, min_obs=4)

        t0 = time.time()
        M_pred = predict_benchpress_scores(M_train)
        elapsed = time.time() - t0
        print(f"  Seed {seed}: predict in {elapsed:.1f}s")

        for i in range(N_MODELS):
            for j in test_cells[i]:
                actual = M_FULL[i, j]
                predicted = M_pred[i, j]
                if np.isfinite(actual) and np.isfinite(predicted):
                    raw_predictions[seed].append((int(i), int(j), float(actual), float(predicted)))

    results = []
    for j in range(N_BENCH):
        bid = BENCH_IDS[j]
        bname = BENCH_NAMES[bid]
        bcat = BENCH_CATS[j]

        # Collect all cells for this benchmark across seeds
        all_actual, all_predicted = [], []
        seed_medapes, seed_medaes = [], []

        for seed in range(N_SEEDS):
            seed_actual, seed_predicted = [], []
            for (mi, bj, act, pred) in raw_predictions[seed]:
                if bj != j:
                    continue
                all_actual.append(act)
                all_predicted.append(pred)
                seed_actual.append(act)
                seed_predicted.append(pred)
            if seed_actual:
                seed_metrics = compute_prediction_error(
                    np.asarray(seed_actual), np.asarray(seed_predicted))
                if np.isfinite(seed_metrics["medape"]):
                    seed_medapes.append(round(seed_metrics["medape"], 2))
                if np.isfinite(seed_metrics["medae"]):
                    seed_medaes.append(round(seed_metrics["medae"], 2))

        metrics = compute_prediction_error(
            np.asarray(all_actual), np.asarray(all_predicted))
        if metrics["n"] < 3 or not np.isfinite(metrics["medape"]):
            print(f"  [{j+1:2d}/{N_BENCH}] {bname}: only {metrics['n']} test cells, skipping")
            continue

        medape = float(metrics["medape"])
        medae = float(metrics["medae"])

        row = {
            'bench_id': bid,
            'bench_name': bname,
            'category': bcat,
            'n_test_cells': metrics["n"],
            'n_seeds': N_SEEDS,
            'medape': round(medape, 2),
            'medae': round(medae, 2),
            'seed_medapes': seed_medapes,
            'seed_medaes': seed_medaes,
        }
        results.append(row)
        print(f"  [{j+1:2d}/{N_BENCH}] {bname:35s}  MedAPE={medape:6.2f}%  "
              f"MedAE={medae:5.2f}  n_cells={metrics['n']}")

    results.sort(key=lambda r: r['medape'])

    # Convert raw_predictions to JSON-serializable format
    raw_for_json = {}
    for seed in range(N_SEEDS):
        raw_for_json[str(seed)] = [
            {'i': t[0], 'j': t[1], 'actual': t[2], 'predicted': t[3]}
            for t in raw_predictions[seed]
        ]

    out = {
        'experiment': 'per_benchmark_predictability',
        'method': 'BenchPress (Logit Bias ALS, rank=2, lambda=0.1)',
        'holdout': 'half_per_model, 10 seeds',
        'n_benchmarks': len(results),
        'results': results,
        'raw_predictions': raw_for_json,
    }

    out_path = os.path.join(SCRIPT_DIR, 'results.json')
    write_json(out_path, out, indent=2)
    print(f"\nSaved {out_path}")

    medapes = [r['medape'] for r in results]
    below_15 = sum(1 for m in medapes if m < 15)
    print(f"\nSummary: {len(results)} benchmarks")
    print(f"  Median MedAPE: {np.median(medapes):.1f}%")
    print(f"  Below 15%: {below_15}/{len(results)} ({below_15/len(results)*100:.0f}%)")


if __name__ == '__main__':
    run()
