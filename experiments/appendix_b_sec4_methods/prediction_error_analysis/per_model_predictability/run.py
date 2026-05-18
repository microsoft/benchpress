#!/usr/bin/env python3
"""
Per-model predictability via half-per-model holdout.

Mirror of `per_benchmark_predictability/run.py` but aggregates by model row
instead of benchmark column. We REUSE the raw predictions stored in
`per_benchmark_predictability/results.json` (same predictor, same holdout,
same 10 seeds) — no re-prediction needed.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import (
    MODEL_IDS, MODEL_NAMES, MODEL_PROVIDERS, N_MODELS,
    compute_prediction_error,
)
from benchpress.artifact_utils import ensure_artifacts
from benchpress.io_utils import load_json, write_json

N_SEEDS = 10

RAW_SOURCE = os.path.abspath(os.path.join(
    REPO_ROOT, 'experiments', 'appendix_c_sec5_findings', 'probe_selection',
    'per_benchmark_predictability', 'results.json'))


def run():
    ensure_artifacts(
        [RAW_SOURCE],
        [
            "{python}",
            os.path.join(os.path.dirname(RAW_SOURCE), "run.py"),
        ],
        description="per-benchmark predictability raw predictions",
    )

    src = load_json(RAW_SOURCE)
    raw = src['raw_predictions']  # dict[str(seed)] -> list of {i,j,actual,predicted}

    seeds_present = sorted(int(s) for s in raw.keys())
    if seeds_present != list(range(N_SEEDS)):
        sys.exit(f"Seed mismatch in canonical raw: got {seeds_present}, "
                 f"expected {list(range(N_SEEDS))}")

    results = []
    for i in range(N_MODELS):
        mid = MODEL_IDS[i]
        mname = MODEL_NAMES[mid]
        provider = str(MODEL_PROVIDERS[i])

        all_actual, all_predicted = [], []
        seed_medapes, seed_medaes = [], []

        for seed in range(N_SEEDS):
            seed_actual, seed_predicted = [], []
            for rec in raw[str(seed)]:
                if rec['i'] != i:
                    continue
                all_actual.append(rec['actual'])
                all_predicted.append(rec['predicted'])
                seed_actual.append(rec['actual'])
                seed_predicted.append(rec['predicted'])
            if seed_actual:
                seed_metrics = compute_prediction_error(
                    np.asarray(seed_actual), np.asarray(seed_predicted))
                if np.isfinite(seed_metrics['medape']):
                    seed_medapes.append(round(seed_metrics['medape'], 2))
                if np.isfinite(seed_metrics['medae']):
                    seed_medaes.append(round(seed_metrics['medae'], 2))

        if not all_actual:
            print(f"  [{i+1:3d}/{N_MODELS}] {mname}: no test cells, skipping")
            continue

        metrics = compute_prediction_error(
            np.asarray(all_actual), np.asarray(all_predicted))
        if metrics['n'] < 3 or not np.isfinite(metrics['medape']):
            print(f"  [{i+1:3d}/{N_MODELS}] {mname}: only {metrics['n']} "
                  f"test cells, skipping")
            continue

        medape = float(metrics['medape'])
        medae = float(metrics['medae'])

        row = {
            'model_id': mid,
            'model_name': mname,
            'provider': provider,
            'n_test_cells': metrics['n'],
            'n_seeds': N_SEEDS,
            'medape': round(medape, 2),
            'medae': round(medae, 2),
            'seed_medapes': seed_medapes,
            'seed_medaes': seed_medaes,
        }
        results.append(row)
        print(f"  [{i+1:3d}/{N_MODELS}] {mname:45s}  MedAPE={medape:6.2f}%  "
              f"MedAE={medae:5.2f}  n_cells={metrics['n']}")

    results.sort(key=lambda r: r['medape'])

    out = {
        'experiment': 'per_model_predictability',
        'method': 'BenchPress (Logit Bias ALS, rank=2, lambda=0.1)',
        'holdout': 'half_per_model, 10 seeds (reused from per_benchmark_predictability)',
        'raw_source': os.path.relpath(RAW_SOURCE, REPO_ROOT),
        'n_models': len(results),
        'results': results,
    }

    out_path = os.path.join(SCRIPT_DIR, 'results.json')
    write_json(out_path, out, indent=2)
    print(f"\nSaved {out_path}")

    medapes = [r['medape'] for r in results]
    below_15 = sum(1 for m in medapes if m < 15)
    print(f"\nSummary: {len(results)} models")
    print(f"  Median MedAPE: {np.median(medapes):.1f}%")
    print(f"  Below 15%: {below_15}/{len(results)} "
          f"({below_15/len(results)*100:.0f}%)")


if __name__ == '__main__':
    run()
