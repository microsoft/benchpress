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

# Nested selection: hyperparameters are chosen on inner validation cells carved
# out of each outer fold's training cells, so no outer test cell takes part in
# selection. The inner partition seed is offset away from the outer seed range
# (base_seed .. base_seed + n_seeds) so the two splits are drawn independently.
INNER_SCORES_DIR = os.path.join(SCRIPT_DIR, 'inner_scores')
NESTED_RESULTS_PATH = os.path.join(SCRIPT_DIR, 'results_nested.json')
INNER_SEED_OFFSET = 4200
N_INNER_FOLDS = 2
INNER_PARTITION_FOLDS = 3

with contextlib.redirect_stdout(io.StringIO()):
    from benchpress.evaluation_harness import (
        M_FULL, load_folds, compute_prediction_error, make_score_predictor,
        holdout_inner_per_model, mask_cells, matrix_identity_sha256,
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
        'matrix_identity_sha256': matrix_identity_sha256(M_FULL),
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
        'matrix_identity_sha256': matrix_identity_sha256(M_FULL),
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


def run_inner_shard(shard_index, n_seeds=10, n_folds=3, base_seed=42, force=False):
    """Score one configuration on the inner validation folds of every outer fold.

    Writes one row per (outer fold, inner fold): the configuration is fit on the
    outer training cells minus that inner validation set and scored on it. Only
    validation scores are persisted, not prediction matrices, because nested
    selection consumes the scores alone; the reported test error still comes from
    the outer predictions written by run_shard().
    """
    shards = all_shards()
    if shard_index < 0 or shard_index >= len(shards):
        raise ValueError(f'shard_index must be in [0, {len(shards)-1}]')
    shard = shards[shard_index]
    path = os.path.join(INNER_SCORES_DIR, f"{shard['shard_id']}.npz")

    if os.path.exists(path) and not force:
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata_json']))
            _validate_shard_metadata(
                metadata, shard, path, n_seeds, n_folds, base_seed)
            print(f"SKIP existing inner shard {shard_index}: {path}")
            return path
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            print(f"Existing inner shard is stale; rerunning: {path}")

    t0 = time.time()
    folds = load_current_folds(n_seeds=n_seeds, n_folds=n_folds,
                               base_seed=base_seed)
    predict_fn = make_score_predictor(
        METHODS[shard['method']], shard['transform'], **shard['hp'])

    rows = []
    for outer_idx, (M_outer_train, outer_test_set) in enumerate(folds):
        outer_seed = base_seed + outer_idx // n_folds
        inner_sets = holdout_inner_per_model(
            outer_test_set, seed=INNER_SEED_OFFSET + outer_seed,
            n_inner_folds=N_INNER_FOLDS,
            n_partition_folds=INNER_PARTITION_FOLDS)
        for inner_idx, inner_val in enumerate(inner_sets):
            M_pred = predict_fn(mask_cells(inner_val, base_matrix=M_outer_train))
            m = compute_prediction_error(
                M_FULL, M_pred, test_set=inner_val, aggregation='pool')
            rows.append((outer_idx, inner_idx, m['medape'], m['medae'],
                         m['n'] / len(inner_val) if inner_val else 0.0,
                         len(inner_val)))

    metadata = {
        **shard,
        'path': os.path.relpath(path, SCRIPT_DIR),
        'n_seeds': n_seeds,
        'n_folds': n_folds,
        'base_seed': base_seed,
        'n_inner_folds': N_INNER_FOLDS,
        'inner_partition_folds': INNER_PARTITION_FOLDS,
        'inner_seed_offset': INNER_SEED_OFFSET,
        'matrix_shape': list(M_FULL.shape),
        'matrix_identity_sha256': matrix_identity_sha256(M_FULL),
        'elapsed_sec': time.time() - t0,
    }

    write_npz_compressed_atomic(
        path,
        rows=np.asarray(rows, dtype=np.float64),
        columns=np.asarray(['outer_idx', 'inner_idx', 'medape', 'medae',
                            'coverage', 'n_cells']),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"WROTE inner shard {shard_index}: {path}")
    print(f"  {shard['transform']} x {shard['method']} hp={shard['hp']}")
    return path


def _finite_median(values):
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float('nan')


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
    fold_sizes = {int(f): int((fold_id == f).sum()) for f in np.unique(fold_id)}
    per_fold = {
        int(f): {
            'medape': float(g['medape']),
            'medae': float(g['medae']),
            'coverage': float(g['n'] / fold_sizes[int(f)])
                        if fold_sizes[int(f)] else 0.0,
        }
        for f, g in metrics['per_group'].items()
    }
    return meta, {
        'medae_median': float(metrics['medae_median']),
        'medape_median': float(metrics['medape_median']),
        'coverage': float(n_cov / n_tot) if n_tot else 0.0,
        'prediction_file': os.path.relpath(path, SCRIPT_DIR),
    }, per_fold


def merge_results():
    shards = all_shards()
    rows = []
    missing = []

    for shard in shards:
        if not os.path.exists(shard['path']):
            run_shard(shard['shard_index'])
            if not os.path.exists(shard['path']):
                missing.append(shard)
                continue
        meta, metrics, _ = _metrics_from_npz(shard['path'], shard)
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
    return results


def merge_nested_results():
    """Leaderboard whose hyperparameters were selected without seeing test cells.

    For each (transform, method) pair and each outer fold, the selected
    hyperparameter is the one with the lowest inner-validation MedAPE on that
    fold; the reported error is that hyperparameter's error on the same outer
    test cells as results.json, summarized as the median over outer folds. Where
    the selection is not unanimous across folds, the row reports the most
    frequently selected hyperparameter together with how often it won.
    """
    shards = all_shards()
    inner_medape, outer_per_fold = {}, {}
    missing_inner, missing_outer, n_outer = [], [], None

    for shard in shards:
        idx = shard['shard_index']
        inner_path = os.path.join(INNER_SCORES_DIR, f"{shard['shard_id']}.npz")
        if not os.path.exists(inner_path):
            missing_inner.append(idx)
            continue
        if not os.path.exists(shard['path']):
            missing_outer.append(idx)
            continue
        with np.load(inner_path, allow_pickle=False) as data:
            rows = data['rows']
        per_outer = {}
        for row in rows:
            per_outer.setdefault(int(row[0]), []).append(float(row[2]))
        inner_medape[idx] = {o: _finite_median(v) for o, v in per_outer.items()}
        meta, _, per_fold = _metrics_from_npz(shard['path'], shard)
        outer_per_fold[idx] = per_fold
        n_outer = int(meta['n_seeds']) * int(meta['n_folds'])

    if missing_inner or missing_outer:
        raise RuntimeError(
            f'{len(missing_inner)} inner shards missing (e.g. '
            f'{missing_inner[:5]}) and {len(missing_outer)} outer prediction '
            f'files missing (e.g. {missing_outer[:5]}); run them before merging')

    pairs = {}
    for shard in shards:
        pairs.setdefault(
            (shard['transform'], shard['method']), []).append(
                shard['shard_index'])

    pair_rows = []
    for (transform, method), candidates in pairs.items():
        picks, medape, medae, coverage = [], [], [], []
        for outer_idx in range(n_outer):
            scored = [i for i in candidates
                      if np.isfinite(inner_medape[i].get(outer_idx, np.nan))]
            if not scored:
                continue
            best = min(scored, key=lambda i: inner_medape[i][outer_idx])
            picks.append(best)
            fold_error = outer_per_fold[best][outer_idx]
            medape.append(fold_error['medape'])
            medae.append(fold_error['medae'])
            coverage.append(fold_error['coverage'])
        if not picks:
            raise RuntimeError(
                f'no inner score for any {transform} x {method} configuration')
        modal = max(set(picks), key=picks.count)
        pair_rows.append({
            'transform': transform,
            'method': method,
            'modal_hp': shards[modal]['hp'],
            'modal_hp_share': picks.count(modal) / len(picks),
            'n_distinct_hp_selected': len(set(picks)),
            'medape_median': _finite_median(medape),
            'medae_median': _finite_median(medae),
            'coverage': float(np.mean(coverage)),
        })

    results = {
        'protocol': {
            'outer': f'{n_outer} folds from results.json, unchanged',
            'inner': f'{N_INNER_FOLDS} of {INNER_PARTITION_FOLDS} per-model '
                     f'partition folds drawn on outer training cells only '
                     f'(seed = {INNER_SEED_OFFSET} + outer seed)',
            'selection_metric': 'MedAPE on inner validation cells',
            'n_candidate_configs': len(shards),
        },
        'pair_rows': sorted(pair_rows, key=lambda r: r['medape_median']),
    }
    write_json(NESTED_RESULTS_PATH, results, indent=2, sort_keys=True)
    print(f"WROTE {NESTED_RESULTS_PATH}")
    print(f"Selected hyperparameters for {len(pair_rows)} pairs "
          f"from {len(shards)} configurations")
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
    parser.add_argument('--inner-shard-index', type=int,
                        help='Score one shard on the inner validation folds and '
                             'write inner_scores/*.npz.')
    parser.add_argument('--merge-nested', action='store_true',
                        help='Re-select hyperparameters on inner validation and '
                             'write results_nested.json.')
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
    if args.inner_shard_index is not None:
        run_inner_shard(args.inner_shard_index, n_seeds=args.n_seeds,
                        n_folds=args.n_folds, base_seed=args.base_seed,
                        force=args.force)
        return
    if args.merge:
        merge_results()
        return
    if args.merge_nested:
        merge_nested_results()
        return
    parser.error('choose --list-shards, --shard-index, --inner-shard-index, '
                 '--merge, or --merge-nested')


if __name__ == '__main__':
    main()
