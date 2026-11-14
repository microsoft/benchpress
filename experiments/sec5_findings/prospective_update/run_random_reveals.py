"""Predict August-only models from the frozen April matrix and exact seed counts.

Run from the repository root with ``python -m
experiments.sec5_findings.prospective_update.run_random_reveals``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing
import os

import numpy as np

from .run import (append_target_row, bootstrap_benchpress, git_commit,
                  load_snapshot, observed_aug_values_for_may_benchmarks)

K_VALUES = (1, 3, 5, 10)
N_SEEDS = 10
BASE_SEED = 42
METHODS = ('benchpress', 'offsets_only', 'model_mean')
APRIL_MATRIX_HASH = '4a0d8ae3dffd6ac9e5d87d242de68066668423120534226cd996b59ebdaafe31'
AUGUST_MATRIX_HASH = '94696a247dddb2eec1c74dcc35518460504ab403fc28349916a11f55b8b7e729'


def predict_unit(reference, benchmark_ids, benchmark_metrics, model_id,
                 target_values, k, seed):
    """Return predictions and revealed benchmarks for one target/count/seed."""
    observed = [bid for bid in benchmark_ids if bid in target_values]
    if len(observed) < k + 5:
        raise ValueError(f'{model_id}: k={k} must leave at least five hidden scores')
    material = f'{BASE_SEED}:{model_id}:{seed}'.encode()
    rng = np.random.RandomState(int(hashlib.sha256(material).hexdigest()[:8], 16))
    rng.shuffle(observed)
    revealed = observed[:k]
    common = set(observed[10:]) if len(observed) >= 15 else set()
    train = append_target_row(
        {'matrix': reference, 'benchmark_ids': benchmark_ids}, target_values, revealed)
    np.testing.assert_equal(train[:-1], reference)
    if np.isfinite(train[-1]).sum() != k:
        raise ValueError(f'{model_id}: expected exactly {k} revealed target scores')

    from benchpress.methods.predictors import (
        predict_benchpress_scores, predict_logit_bias_als_scores,
        predict_logit_model_mean_scores,
    )
    kwargs = {'metric': benchmark_metrics, 'benchmark_ids': benchmark_ids}
    predictions = {
        'benchpress': predict_benchpress_scores(train.copy(), **kwargs)[-1],
        'offsets_only': predict_logit_bias_als_scores(
            train.copy(), rank=0, lam=0.1, **kwargs)[-1],
        'model_mean': predict_logit_model_mean_scores(train.copy(), **kwargs)[-1],
    }
    columns = {bid: j for j, bid in enumerate(benchmark_ids)}
    rows = []
    for bid in observed[k:]:
        j = columns[bid]
        values = {method: float(predictions[method][j]) for method in METHODS}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f'{model_id}/{bid}: non-finite prediction: {values}')
        rows.append({
            'model_id': model_id, 'k': k, 'seed': seed,
            'benchmark_id': bid, 'actual': float(target_values[bid]),
            'common_test': bid in common, **values,
        })
    return {'model_id': model_id, 'k': k, 'seed': seed,
            'n_observed_scores': len(observed),
            'revealed_benchmarks': revealed, 'rows': rows}


def summarize(units):
    """Return pooled and model-level errors on all hidden and fixed test cells."""
    from benchpress.evaluation_harness import compute_prediction_error

    summaries = []
    for scope in ('all_eligible', 'matched_targets_and_cells'):
        for k in K_VALUES:
            selected = [unit for unit in units if unit['k'] == k]
            rows = [row for unit in selected for row in unit['rows']
                    if scope == 'all_eligible' or row['common_test']]
            if not rows:
                continue
            for method in METHODS:
                metrics = compute_prediction_error(
                    [r['actual'] for r in rows], [r[method] for r in rows],
                    aggregation='pool')
                per_model = {}
                for unit in selected:
                    subset = [r for r in unit['rows']
                              if scope == 'all_eligible' or r['common_test']]
                    if subset:
                        score = compute_prediction_error(
                            [r['actual'] for r in subset], [r[method] for r in subset],
                            aggregation='pool')
                        per_model.setdefault(unit['model_id'], []).append(float(score['medae']))
                summaries.append({
                    'scope': scope, 'k': k, 'method': method,
                    'n_models': len(per_model), 'n_prediction_records': len(rows),
                    'n_unique_cells': len({(r['model_id'], r['benchmark_id']) for r in rows}),
                    'medae': float(metrics['medae']), 'medape': float(metrics['medape']),
                    'median_model_medae': float(np.median([
                        np.median(values) for values in per_model.values()])),
                    'per_model_medae': {mid: float(np.median(values))
                                        for mid, values in per_model.items()},
                })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-json', required=True)
    parser.add_argument('--eval-json', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--mode', choices=('check', 'run', 'summarize'), default='run')
    args = parser.parse_args()
    bootstrap_benchpress(args.train_json)
    from benchpress.build_benchmark_matrix import load_score_matrix
    from benchpress.evaluation_harness import matrix_identity_sha256
    from benchpress.io_utils import load_json, safe_token, write_json_atomic

    train = load_snapshot(args.train_json, load_score_matrix)
    august = load_snapshot(args.eval_json, load_score_matrix)
    if matrix_identity_sha256(train['matrix']) != APRIL_MATRIX_HASH:
        raise ValueError('Training matrix does not match the archived 84 x 133 April matrix')
    if matrix_identity_sha256(august['matrix']) != AUGUST_MATRIX_HASH:
        raise ValueError('Evaluation matrix does not match the archived August matrix')
    targets = {
        mid: observed_aug_values_for_may_benchmarks(august, mid, train['benchmark_ids'])
        for mid in august['model_ids'] if mid not in set(train['model_ids'])
    }
    target_inventory = [
        {'model_id': mid, 'n_main_benchmark_scores': len(values),
         'eligible_k': [k for k in K_VALUES if len(values) >= k + 5]}
        for mid, values in sorted(targets.items())
    ]
    file_hashes = {}
    for key, path in (('train', args.train_json), ('evaluation', args.eval_json)):
        with open(path, 'rb') as handle:
            file_hashes[key] = hashlib.sha256(handle.read()).hexdigest()
    config = {
        'git_commit': git_commit(), 'input_sha256': file_hashes,
        'train_matrix_sha256': APRIL_MATRIX_HASH, 'evaluation_matrix_sha256': AUGUST_MATRIX_HASH,
        'train_shape': list(train['matrix'].shape), 'train_n_observed': 2604,
        'train_model_ids': train['model_ids'], 'benchmark_ids': train['benchmark_ids'],
        'target_inventory': target_inventory, 'base_seed': BASE_SEED,
        'n_seeds': N_SEEDS, 'k_values': list(K_VALUES), 'methods': list(METHODS),
        'predictor': 'Logit Bias ALS rank 2, lambda 0.1; offsets only uses rank 0',
        'training_rule': 'Frozen April rows plus the current target revealed scores only',
        'target_rule': 'August filtered-matrix rows absent from the April filtered matrix',
        'benchmark_rule': 'April matrix columns only',
        'inclusion_rule': 'At least k + 5 observed scores on April benchmarks',
        'matched_rule': 'Targets with at least 15 scores; test outside first 10 reveals for each seed',
    }
    import json
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = os.path.join(args.out_dir, 'manifest.json')
    if os.path.exists(config_path) and load_json(config_path)['config_sha256'] != identity:
        raise ValueError(f'{config_path}: existing result belongs to a different configuration')
    os.makedirs(os.path.join(args.out_dir, 'units'), exist_ok=True)
    write_json_atomic(config_path, {'config_sha256': identity, **config}, indent=2)
    tasks = [(mid, k, seed) for mid, values in sorted(targets.items())
             for k in K_VALUES if len(values) >= k + 5 for seed in range(N_SEEDS)]
    planned = tasks[:1] if args.mode == 'check' else tasks
    print(f'Targets={len(targets)} planned_units={len(tasks)} mode={args.mode}', flush=True)
    units, pending = [], []
    for mid, k, seed in planned:
        path = os.path.join(args.out_dir, 'units', f'{safe_token(mid)}__k{k}__s{seed}.json.gz')
        if os.path.exists(path):
            unit = load_json(path)
            if unit['config_sha256'] != identity or (unit['model_id'], unit['k'], unit['seed']) != (mid, k, seed):
                raise ValueError(f'{path}: cached unit identity mismatch')
            units.append(unit)
        else:
            pending.append((mid, k, seed, path))
    if pending and args.mode == 'summarize':
        raise FileNotFoundError(f'{len(pending)} expected units are missing')
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context('fork')) as pool:
            futures = {
                pool.submit(predict_unit, train['matrix'], train['benchmark_ids'],
                            train['benchmark_metrics'], mid, targets[mid], k, seed): path
                for mid, k, seed, path in pending
            }
            for future in as_completed(futures):
                unit = future.result()
                unit['config_sha256'] = identity
                write_json_atomic(futures[future], unit, indent=2)
                units.append(unit)
                if len(units) % 25 == 0 or len(units) == len(planned):
                    print(f'Completed {len(units)}/{len(planned)} units', flush=True)
    units.sort(key=lambda unit: (unit['model_id'], unit['k'], unit['seed']))
    summary = {'config_sha256': identity, 'complete': len(units) == len(tasks),
               'n_completed_units': len(units), 'n_expected_units': len(tasks),
               'target_inventory': target_inventory, 'results': summarize(units)}
    name = 'check_summary.json' if args.mode == 'check' else 'summary.json'
    write_json_atomic(os.path.join(args.out_dir, name), summary, indent=2)
    if args.mode != 'check':
        write_json_atomic(os.path.join(args.out_dir, 'raw_predictions.json.gz'),
                          [row for unit in units for row in unit['rows']])
    print(f'Wrote {os.path.join(args.out_dir, name)}', flush=True)


if __name__ == '__main__':
    main()
