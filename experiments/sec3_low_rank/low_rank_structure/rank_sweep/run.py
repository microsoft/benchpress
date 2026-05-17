#!/usr/bin/env python3
"""Rank sweep for Soft-Impute SVD in raw and logit score spaces.

This is the main-body Sec. 3.3 prediction evidence: no BenchPress blend
reference, only Soft-Impute over raw and transformed benchmark scores.
"""
import os, sys, warnings, numpy as np
from json import JSONDecodeError
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, ROOT)

from benchpress.evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    compute_prediction_error, load_folds, make_score_predictor
)
from benchpress.io_utils import load_json, write_json
from benchpress.methods.completers import (
    complete_soft_impute,
)

SEED = 42
N_SEEDS = 10
N_FOLDS = 3
MIN_SCORES = 1
RANKS = list(range(1, 11))  # 1..10

METHODS = {
    'identity_svd': {
        'predict_fn': complete_soft_impute,
        'extra_kwargs': {},
        'label': 'Raw-space Soft-Impute',
    },
    'logit_svd': {
        'predict_fn': lambda M_train, rank: make_score_predictor(
            complete_soft_impute, 'logit', rank=rank, normalize=False)(M_train),
        'extra_kwargs': {},
        'label': 'Logit-space Soft-Impute',
    },
}


def evaluate_method(predict_fn, folds, **kwargs):
    """Run predict_fn on each fold and keep held-out predictions."""
    all_true, all_pred = [], []
    predictions = []

    for fold_idx, (M_train, test_set) in enumerate(folds):
        seed_idx = fold_idx // N_FOLDS
        fold_in_seed = fold_idx % N_FOLDS
        M_pred = predict_fn(M_train, **kwargs)
        for i, j in test_set:
            t, p = M_FULL[i, j], M_pred[i, j]
            if np.isfinite(t) and np.isfinite(p):
                all_true.append(t)
                all_pred.append(p)
                predictions.append({
                    'seed': int(seed_idx),
                    'fold': int(fold_in_seed),
                    'model_idx': int(i),
                    'benchmark_idx': int(j),
                    'true': float(t),
                    'pred': float(p),
                })

    # Pooled vector metrics
    actual_arr = np.array(all_true)
    pred_arr = np.array(all_pred)
    vec_metrics = compute_prediction_error(actual_arr, pred_arr)

    return {
        'medape': vec_metrics['medape'],
        'medae': vec_metrics['medae'],
        'raw_predictions': predictions,
    }


def load_current_folds():
    """Load canonical folds; fail loudly rather than rewriting them."""
    try:
        return load_folds(n_seeds=N_SEEDS, n_folds=N_FOLDS,
                          base_seed=SEED, min_scores=MIN_SCORES)
    except AssertionError as exc:
        raise RuntimeError(
            "Canonical folds do not match the current score matrix. "
            "Do not regenerate them from this experiment; update the shared "
            "fold artifact through the evaluation workflow first."
        ) from exc


def main():
    np.random.seed(SEED)
    folds = load_current_folds()
    print(f"Matrix: {N_MODELS} models × {N_BENCH} benchmarks; {len(folds)} folds")

    results = {
        'ranks': RANKS,
        'methods': list(METHODS.keys()),
        'matrix': {
            'n_models': int(N_MODELS),
            'n_benchmarks': int(N_BENCH),
            'n_observed': int(np.isfinite(M_FULL).sum()),
            'fill_rate': float(np.isfinite(M_FULL).mean()),
            'model_ids': MODEL_IDS,
            'benchmark_ids': BENCH_IDS,
            'filter': {
                'm_threshold': 15,
                'b_threshold': 8,
                'statuses': ['verified', 'verified_third_party'],
            },
        },
        'folds': {
            'n_seeds': N_SEEDS,
            'n_folds': N_FOLDS,
            'base_seed': SEED,
            'min_scores': MIN_SCORES,
        },
    }
    for key in METHODS:
        results[key] = {}

    out = os.path.join(HERE, 'results.json')

    # Resume support: if results.json exists, load existing per-(method, rank) entries
    if os.path.exists(out):
        try:
            prev = load_json(out)
            for key in METHODS:
                if key in prev and isinstance(prev[key], dict):
                    for r_str, entry in prev[key].items():
                        if {'medape', 'medae'}.issubset(entry):
                            if 'raw_predictions' in entry:
                                results[key][r_str] = entry
            print(f"Loaded {sum(len(v) for v in (results[k] for k in METHODS))} cached entries from {out}")
        except (OSError, JSONDecodeError) as exc:
            print(f"Could not resume from {out}: {exc}")

    # Sweep each method × each rank, saving after every entry for resumability
    for key, spec in METHODS.items():
        predict_fn = spec['predict_fn']
        extra = spec['extra_kwargs']
        label = spec['label']
        for r in RANKS:
            r_key = str(r)
            if r_key in results[key]:
                cached = results[key][r_key]
                print(f"{label} rank={r}: cached MedAPE={cached['medape']:.2f}%  MedAE={cached['medae']:.2f}")
                continue
            m = evaluate_method(predict_fn, folds, rank=r, **extra)
            results[key][r_key] = {
                'medape': m['medape'], 'medae': m['medae'],
                'raw_predictions': m['raw_predictions'],
            }
            write_json(out, results, indent=2)
            print(f"{label} rank={r}: MedAPE={m['medape']:.2f}%  MedAE={m['medae']:.2f}")

    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
