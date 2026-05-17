#!/usr/bin/env python3
"""
Section 4.2 transform x method grid.

This runner is prediction-first: each shard writes the full per-fold prediction
matrices to predictions/*.npz. Metrics in results.json are derived artifacts and
can be regenerated without rerunning predictors.
"""

import argparse
import contextlib
import io
import json
import os
import time
import warnings

import numpy as np
from benchpress.io_utils import write_json, write_npz_compressed_atomic
from benchpress.shard_utils import hp_short_hash, slug

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(SCRIPT_DIR, 'predictions')
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'results.json')
MANIFEST_PATH = os.path.join(SCRIPT_DIR, 'manifest.json')

with contextlib.redirect_stdout(io.StringIO()):
    from benchpress.evaluation_harness import (
        M_FULL, load_folds, compute_prediction_error, make_score_predictor,
    )
    from benchpress.all_methods import (
        complete_benchmark_mean, complete_model_mean,
        complete_model_knn, complete_bench_knn,
        complete_benchreg, complete_modelreg, complete_soft_impute, complete_bias_als,
        complete_mlp, complete_nmf, complete_pmf, complete_nuclear_norm,
        TRANSFORMS,
    )


# Pipeline methods use normalize=False where applicable because data is already
# transformed and z-scored by make_score_predictor().
METHODS = {
    'Benchmark Mean':  complete_benchmark_mean,
    'Model Mean':      complete_model_mean,
    'Bench-KNN':       complete_bench_knn,
    'Model-KNN':       complete_model_knn,
    'BenchReg':        complete_benchreg,
    'ModelReg':        complete_modelreg,
    'Soft-Impute':     lambda M, **kw: complete_soft_impute(M, normalize=False, **kw),
    'Bias ALS':        lambda M, **kw: complete_bias_als(M, normalize=False, **kw),
    'NMF':             lambda M, **kw: complete_nmf(M, normalize=False, **kw),
    'PMF':             lambda M, **kw: complete_pmf(M, normalize=False, **kw),
    'Nuclear Norm':    lambda M, **kw: complete_nuclear_norm(M, normalize=False, **kw),
    'MLP':             complete_mlp,
}

HP_GRIDS = {
    'Benchmark Mean': [{}],
    'Model Mean':     [{}],
    'Bench-KNN':      [{'k': k} for k in [3, 5, 7, 10]],
    'Model-KNN':      [{'k': k} for k in [3, 5, 7, 10]],
    'BenchReg':       [{'top_k': k, 'min_r2': r2}
                       for k in [3, 5, 7]
                       for r2 in [0.1, 0.2, 0.3]],
    'ModelReg':       [{'top_k': k, 'min_r2': r2}
                       for k in [3, 5, 7]
                       for r2 in [0.1, 0.2, 0.3]],
    'Soft-Impute':    [{'rank': 2}],
    'Bias ALS':       [{'rank': 2, 'lam': lam}
                       for lam in [0.01, 0.1, 1.0]],
    'NMF':            [{'rank': r} for r in [1, 2, 3, 5]],
    'PMF':            [{'rank': r} for r in [1, 2, 3, 5]],
    'Nuclear Norm':   [{'lam': l} for l in [0.1, 0.5, 1.0, 5.0]],
    'MLP':            [{'lr': lr} for lr in [1e-4, 1e-3, 1e-2]],
}


def all_shards():
    shards = []
    for tname in TRANSFORMS:
        for mname in METHODS:
            for hp_index, hp in enumerate(HP_GRIDS[mname]):
                shard_index = len(shards)
                shard_id = (
                    f"{shard_index:04d}__{slug(tname)}__"
                    f"{slug(mname)}__hp{hp_index:02d}_{hp_short_hash(hp)}"
                )
                shards.append({
                    'shard_index': shard_index,
                    'shard_id': shard_id,
                    'transform': tname,
                    'method': mname,
                    'hp_index': hp_index,
                    'hp': hp,
                    'path': os.path.join(PRED_DIR, f'{shard_id}.npz'),
                })
    return shards


def _validate_shard_metadata(meta, shard, path, n_seeds, n_folds, base_seed):
    expected = {
        'shard_index': shard['shard_index'],
        'shard_id': shard['shard_id'],
        'transform': shard['transform'],
        'method': shard['method'],
        'hp_index': shard['hp_index'],
        'hp': shard['hp'],
        'path': os.path.relpath(path, SCRIPT_DIR),
        'n_seeds': n_seeds,
        'n_folds': n_folds,
        'base_seed': base_seed,
        'matrix_shape': list(M_FULL.shape),
    }
    actual = {key: meta.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Shard metadata mismatch for {path}: "
            f"expected {expected}, found {actual}"
        )


def _load_completed_npz(path, shard, n_seeds, n_folds, base_seed):
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {'M_pred_by_fold', 'fold_id', 'test_i', 'test_j',
                        'actual', 'predicted', 'metadata_json'}
            if not required.issubset(set(data.files)):
                return None
            out = {k: data[k] for k in data.files}
            _validate_shard_metadata(
                json.loads(str(out['metadata_json'])), shard, path,
                n_seeds, n_folds, base_seed,
            )
            return out
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def load_current_folds(n_seeds, n_folds, base_seed, min_scores=1):
    """Load the persisted fold protocol; fail fast if it no longer matches."""
    return load_folds(
        n_seeds=n_seeds, n_folds=n_folds,
        base_seed=base_seed, min_scores=min_scores,
    )


def run_shard(shard_index, n_seeds=10, n_folds=3, base_seed=42, force=False):
    shards = all_shards()
    if shard_index < 0 or shard_index >= len(shards):
        raise ValueError(f'shard_index must be in [0, {len(shards)-1}]')
    shard = shards[shard_index]

    if os.path.exists(shard['path']) and not force:
        existing = _load_completed_npz(
            shard['path'], shard, n_seeds, n_folds, base_seed)
        if existing is not None:
            print(f"SKIP existing shard {shard_index}: {shard['path']}")
            return shard['path']
        print(f"Existing shard file is invalid; rerunning: {shard['path']}")

    t0 = time.time()
    folds = load_current_folds(n_seeds=n_seeds, n_folds=n_folds,
                               base_seed=base_seed)
    method_fn = METHODS[shard['method']]
    predict_fn = make_score_predictor(
        method_fn, shard['transform'], **shard['hp'])

    pred_mats, fold_ids, test_i, test_j, actual, predicted = [], [], [], [], [], []
    for fold_id, (M_train, test_set) in enumerate(folds):
        M_pred = predict_fn(M_train)
        pred_mats.append(M_pred.astype(np.float64, copy=False))
        for i, j in test_set:
            fold_ids.append(fold_id)
            test_i.append(i)
            test_j.append(j)
            actual.append(M_FULL[i, j])
            predicted.append(M_pred[i, j])

    metadata = {
        **shard,
        'path': os.path.relpath(shard['path'], SCRIPT_DIR),
        'n_seeds': n_seeds,
        'n_folds': n_folds,
        'base_seed': base_seed,
        'matrix_shape': list(M_FULL.shape),
        'elapsed_sec': time.time() - t0,
    }

    write_npz_compressed_atomic(
        shard['path'],
        M_pred_by_fold=np.stack(pred_mats, axis=0),
        fold_id=np.asarray(fold_ids, dtype=np.int16),
        test_i=np.asarray(test_i, dtype=np.int16),
        test_j=np.asarray(test_j, dtype=np.int16),
        actual=np.asarray(actual, dtype=np.float64),
        predicted=np.asarray(predicted, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"WROTE shard {shard_index}: {shard['path']}")
    print(f"  {shard['transform']} x {shard['method']} hp={shard['hp']}")
    return shard['path']


def _metrics_from_npz(path, expected_shard):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json']))
        _validate_shard_metadata(meta, expected_shard, path, 10, 3, 42)
        pred_mats = data['M_pred_by_fold']
        fold_id = data['fold_id'].astype(int)
        test_i = data['test_i'].astype(int)
        test_j = data['test_j'].astype(int)

    M_pred_by_fold = {int(k): pred_mats[int(k)] for k in np.unique(fold_id)}
    test_set = list(zip(test_i.tolist(), test_j.tolist()))
    groups = fold_id.tolist()
    metrics = compute_prediction_error(
        M_FULL, M_pred_by_fold, test_set=test_set, groups=groups,
        aggregation='per_group_median')
    n_cov = sum(g['n'] for g in metrics['per_group'].values())
    n_tot = len(test_set)
    return meta, {
        'medae_median': float(metrics['medae_median']),
        'medape_median': float(metrics['medape_median']),
        'coverage': float(n_cov / n_tot) if n_tot else 0.0,
        'prediction_file': os.path.relpath(path, SCRIPT_DIR),
    }


def merge_results():
    shards = all_shards()
    by_shard_id = {s['shard_id']: s for s in shards}
    rows, missing = [], []

    for shard in shards:
        if not os.path.exists(shard['path']):
            missing.append(shard)
            continue
        meta, metrics = _metrics_from_npz(shard['path'], shard)
        row = {
            'shard_index': int(meta['shard_index']),
            'shard_id': meta['shard_id'],
            'transform': meta['transform'],
            'method': meta['method'],
            'hp_index': int(meta['hp_index']),
            'hp': meta['hp'],
            **metrics,
        }
        rows.append(row)

    results = {}
    for row in rows:
        tname, mname = row['transform'], row['method']
        results.setdefault(tname, {})
        current = results[tname].get(mname)
        if current is None or row['medape_median'] < current['medape_median']:
            results[tname][mname] = {
                'medae_median': row['medae_median'],
                'medape_median': row['medape_median'],
                'coverage': row['coverage'],
                'best_hp': row['hp'],
                'best_hp_index': row['hp_index'],
                'prediction_file': row['prediction_file'],
            }

    manifest = {
        'n_total_shards': len(shards),
        'n_completed_shards': len(rows),
        'n_missing_shards': len(missing),
        'missing_shards': [
            {k: v for k, v in s.items() if k != 'path'} for s in missing
        ],
        'completed': rows,
        'expected': [
            {**{k: v for k, v in s.items() if k != 'path'},
             'path': os.path.relpath(s['path'], SCRIPT_DIR)}
            for s in shards
        ],
    }

    write_json(RESULTS_PATH, results, indent=2, sort_keys=True)
    write_json(MANIFEST_PATH, manifest, indent=2, sort_keys=True)

    print(f"WROTE {RESULTS_PATH}")
    print(f"WROTE {MANIFEST_PATH}")
    print(f"Completed {len(rows)}/{len(shards)} shards")
    if missing:
        print("Missing shard indices:", ', '.join(str(s['shard_index']) for s in missing))
    return results


def print_shards():
    for shard in all_shards():
        done = os.path.exists(shard['path'])
        print(json.dumps({
            'shard_index': shard['shard_index'],
            'done': done,
            'transform': shard['transform'],
            'method': shard['method'],
            'hp_index': shard['hp_index'],
            'hp': shard['hp'],
            'path': os.path.relpath(shard['path'], SCRIPT_DIR),
        }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-shards', action='store_true',
                        help='Print all transform x method x HP shards as JSONL.')
    parser.add_argument('--shard-index', type=int,
                        help='Run one shard and write predictions/*.npz.')
    parser.add_argument('--merge', action='store_true',
                        help='Recompute metrics from predictions/*.npz.')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite an existing shard npz.')
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--n-folds', type=int, default=3)
    parser.add_argument('--base-seed', type=int, default=42)
    args = parser.parse_args()

    if args.list_shards:
        print_shards()
        return
    if args.shard_index is not None:
        run_shard(args.shard_index, n_seeds=args.n_seeds,
                  n_folds=args.n_folds, base_seed=args.base_seed,
                  force=args.force)
        return
    if args.merge:
        merge_results()
        return
    parser.error('choose --list-shards, --shard-index, or --merge')


if __name__ == '__main__':
    main()
