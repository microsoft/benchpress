#!/usr/bin/env python
"""Random probe baseline under the model-split validation protocol.

For each seed, draw one global random benchmark ordering. For each k, validate
the first k probes on held-out model rows using the same isolated-new-model
protocol as run_model_split_validation.py.
"""

import argparse
import os
import random
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.all_methods import predict_benchpress_scores
from benchpress.evaluation_harness import (
    BENCH_IDS,
    MODEL_IDS,
    MODEL_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    evaluate_probe_set_on_heldout_models,
    pack_probe_predictions,
    random_global_probe_set,
    split_models_for_probe_validation,
)
from benchpress.io_utils import load_json, write_json_atomic
from benchpress.shard_utils import (
    default_k_seed_shard_name,
    merge_prediction_shards,
    run_k_seed_shards,
    validate_k_seed_result_payload,
    write_prediction_result,
)


SEED = 42
PROTOCOL = 'model_split_random_probe_validation_v1'
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DEFAULT_OUT = os.path.join(RESULTS_DIR, 'model_split_random_medae_train70.json.gz')
DEFAULT_SHARD_DIR = os.path.join(RESULTS_DIR, 'model_split_random_medae_train70_shards')
K_MAX = 10
N_SEEDS = 10
TRAIN_FRACTION = 0.7

np.random.seed(SEED)
random.seed(SEED)


def _metric_payload(metrics):
    return {
        'medape': float(metrics['medape']) if np.isfinite(metrics['medape']) else None,
        'medae': float(metrics['medae']) if np.isfinite(metrics['medae']) else None,
        'n': int(metrics['n']),
    }


def _model_id_payload(indices):
    return [MODEL_IDS[int(i)] for i in indices]


def _model_name_payload(indices):
    return {
        MODEL_IDS[int(i)]: MODEL_NAMES.get(MODEL_IDS[int(i)], MODEL_IDS[int(i)])
        for i in indices
    }


def _probe_ids(probe_set):
    return [BENCH_IDS[int(j)] for j in sorted(probe_set)]


def run_one(k, seed_idx, train_fraction=TRAIN_FRACTION, split_seed=SEED, model_limit=None):
    split_model_indices = None
    if model_limit is not None:
        split_model_indices = list(range(min(N_MODELS, int(model_limit))))
    train_model_indices, validation_model_indices = split_models_for_probe_validation(
        train_fraction=train_fraction,
        seed=split_seed,
        model_indices=split_model_indices,
    )
    probe_set = random_global_probe_set(k, seed_idx, base_seed=SEED)
    start = time.time()
    predictions, metrics, score = evaluate_probe_set_on_heldout_models(
        probe_set,
        predict_benchpress_scores,
        validation_model_indices,
        train_model_indices,
        metric='medae',
        include_probe_targets=False,
    )
    raw = []
    for i, j, actual, pred in sorted(predictions, key=lambda p: (p[0], p[1])):
        if np.isfinite(actual) and np.isfinite(pred):
            raw.append({
                'seed': int(seed_idx),
                'k': int(k),
                'model': int(i),
                'bench': int(j),
                'actual': round(float(actual), 6),
                'pred': round(float(pred), 6),
            })
    return {
        'raw': raw,
        'probe_ids': _probe_ids(probe_set),
        'metrics': _metric_payload(metrics),
        'score': float(score) if np.isfinite(score) else None,
        'elapsed_s': time.time() - start,
        'train_model_indices': [int(i) for i in train_model_indices],
        'validation_model_indices': [int(i) for i in validation_model_indices],
    }


def _base_config(k_max=K_MAX, n_seeds=N_SEEDS, train_fraction=TRAIN_FRACTION,
                 split_seed=SEED, model_limit=None):
    split_model_indices = None
    if model_limit is not None:
        split_model_indices = list(range(min(N_MODELS, int(model_limit))))
    train_model_indices, validation_model_indices = split_models_for_probe_validation(
        train_fraction=train_fraction,
        seed=split_seed,
        model_indices=split_model_indices,
    )
    return {
        'protocol': PROTOCOL,
        'metric': 'medae',
        'k_max': int(k_max),
        'n_seeds': int(n_seeds),
        'base_seed': SEED,
        'train_fraction': float(train_fraction),
        'split_seed': int(split_seed),
        'model_limit': model_limit,
        'n_models': N_MODELS,
        'n_bench': N_BENCH,
        'n_observed': int(OBSERVED.sum()),
        'n_train_models': len(train_model_indices),
        'n_validation_models': len(validation_model_indices),
        'train_model_ids': _model_id_payload(train_model_indices),
        'validation_model_ids': _model_id_payload(validation_model_indices),
        'train_model_names': _model_name_payload(train_model_indices),
        'validation_model_names': _model_name_payload(validation_model_indices),
        'bench_ids': BENCH_IDS,
        'validation_eval_scope': (
            'held-out model rows; non-probe metrics exclude already measured '
            'probe cells'
        ),
        'cell_masking': (
            'For each seed, choose one global random benchmark ordering. For '
            'each k, use the first k columns from that ordering. Validation '
            'isolates each held-out target model: the predictor sees training '
            'rows plus that target model probe cells, and does not see other '
            'held-out model rows.'
        ),
        'prediction_engine': (
            'predict_benchpress_scores (Logit Bias ALS, rank=2, lambda=0.1)'
        ),
    }


def write_random_split_result(raw_predictions, output_path, probe_sets,
                              k_max=K_MAX, n_seeds=N_SEEDS,
                              train_fraction=TRAIN_FRACTION, split_seed=SEED,
                              model_limit=None, indent=None):
    output = write_prediction_result(
        raw_predictions,
        output_path,
        _base_config(
            k_max=k_max,
            n_seeds=n_seeds,
            train_fraction=train_fraction,
            split_seed=split_seed,
            model_limit=model_limit,
        ),
        include_summary_by_k_seed=True,
        indent=indent,
    )
    output['probe_sets'] = probe_sets
    write_json_atomic(output_path, output, indent=indent)
    return output


def _expected_config(k, seed_idx, train_fraction, split_seed, model_limit):
    return {
        'protocol': PROTOCOL,
        'metric': 'medae',
        'k_max': int(k),
        'n_seeds': 1,
        'base_seed': SEED,
        'train_fraction': float(train_fraction),
        'split_seed': int(split_seed),
        'model_limit': model_limit,
        'n_models': N_MODELS,
        'n_bench': N_BENCH,
    }


def _expected_shard_name(k, seed_idx, model_limit):
    suffix = f'_m{model_limit}' if model_limit is not None else ''
    return default_k_seed_shard_name(k, seed_idx, suffix=suffix, extension='.json.gz')


def _validate_shard_payload(data, path, expected_k, expected_seed,
                            train_fraction=TRAIN_FRACTION, split_seed=SEED,
                            model_limit=None):
    validate_k_seed_result_payload(
        data,
        path,
        expected_k,
        expected_seed,
        expected_config=_expected_config(
            expected_k, expected_seed, train_fraction, split_seed, model_limit,
        ),
        require_summary=True,
    )


def write_single_shard(k, seed_idx, output_path, train_fraction=TRAIN_FRACTION,
                       split_seed=SEED, model_limit=None):
    print(f"Single shard: k={k} seed={seed_idx} -> {output_path}", flush=True)
    result = run_one(
        k,
        seed_idx,
        train_fraction=train_fraction,
        split_seed=split_seed,
        model_limit=model_limit,
    )
    print(
        f"k={k} seed={seed_idx}: score={result['score']:.3f} "
        f"n={result['metrics']['n']} rows={len(result['raw'])} "
        f"{result['elapsed_s']:.0f}s",
        flush=True,
    )
    write_random_split_result(
        result['raw'],
        output_path,
        probe_sets={f'k{k}_seed{seed_idx}': result['probe_ids']},
        k_max=k,
        n_seeds=1,
        train_fraction=train_fraction,
        split_seed=split_seed,
        model_limit=model_limit,
    )


def merge_shards(shard_dir, output_path, k_max, n_seeds,
                 train_fraction=TRAIN_FRACTION, split_seed=SEED, model_limit=None):
    expected_names = {
        _expected_shard_name(k, seed_idx, model_limit)
        for k in range(1, k_max + 1)
        for seed_idx in range(n_seeds)
    }
    raw = merge_prediction_shards(
        shard_dir,
        expected_names,
        _validate_shard_payload,
        validation_kwargs={
            'train_fraction': train_fraction,
            'split_seed': split_seed,
            'model_limit': model_limit,
        },
    )
    probe_sets = {}
    for name in sorted(expected_names):
        payload = load_json(os.path.join(shard_dir, name))
        probe_sets.update(payload.get('probe_sets', {}))
    write_random_split_result(
        raw,
        output_path,
        probe_sets=probe_sets,
        k_max=k_max,
        n_seeds=n_seeds,
        train_fraction=train_fraction,
        split_seed=split_seed,
        model_limit=model_limit,
        indent=2,
    )


def _parse_k_values(spec, k_max):
    if spec:
        values = []
        for item in spec.split(','):
            item = item.strip()
            if '-' in item:
                lo, hi = item.split('-', 1)
                values.extend(range(int(lo), int(hi) + 1))
            elif item:
                values.append(int(item))
        return sorted(set(values))
    return list(range(1, k_max + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-max', type=int, default=K_MAX)
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--k-values', type=str, default=None)
    parser.add_argument('--train-fraction', type=float, default=TRAIN_FRACTION)
    parser.add_argument('--split-seed', type=int, default=SEED)
    parser.add_argument('--model-limit', type=int, default=None)
    parser.add_argument('--output', type=str, default=DEFAULT_OUT)
    parser.add_argument('--shard-dir', type=str, default=DEFAULT_SHARD_DIR)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    k_values = _parse_k_values(args.k_values, args.k_max)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.merge:
        merge_shards(
            args.shard_dir,
            args.output,
            args.k_max,
            args.n_seeds,
            train_fraction=args.train_fraction,
            split_seed=args.split_seed,
            model_limit=args.model_limit,
        )
        return

    run_k_seed_shards(
        k_values,
        args.n_seeds,
        args.shard_dir,
        args.workers,
        shard_name_fn=lambda k, seed_idx: _expected_shard_name(
            k, seed_idx, args.model_limit,
        ),
        write_shard_fn=write_single_shard,
        validate_shard_fn=_validate_shard_payload,
        job_kwargs={
            'train_fraction': args.train_fraction,
            'split_seed': args.split_seed,
            'model_limit': args.model_limit,
        },
        label_extra=f"train_fraction={args.train_fraction} split_seed={args.split_seed}",
    )
    merge_shards(
        args.shard_dir,
        args.output,
        args.k_max,
        args.n_seeds,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        model_limit=args.model_limit,
    )


if __name__ == '__main__':
    main()
