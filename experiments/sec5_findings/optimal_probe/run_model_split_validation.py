#!/usr/bin/env python3
"""Model-split validation for greedy BenchPress probe selection.

This script selects probe benchmarks on a training split of model rows and
validates each selected prefix on held-out model rows. Validation uses an
isolated-new-model protocol: for each held-out target model, the predictor sees
the full training-model score matrix plus that target model's observed probe
scores, but not the other held-out model rows.
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.all_methods import predict_benchpress_scores
from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_NAMES,
    MODEL_IDS,
    MODEL_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    base_matrix_for_model_context,
    evaluate_probe_set,
    evaluate_probe_set_on_heldout_models,
    load_benchmark_allowlist,
    pack_probe_predictions,
    probe_candidate_cache_path,
    split_models_for_probe_validation,
    target_by_model_from_models,
)
from benchpress.io_utils import load_json, safe_token, write_json_atomic
from benchpress.shard_utils import short_text_hash

SEED = 42
MODEL_SPLIT_PROTOCOL = 'model_split_probe_validation_v1'
np.random.seed(SEED)
random.seed(SEED)

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_STEPS = 10
TRAIN_FRACTION = 0.7

SCORE_UNIT = {
    'medape': '%',
    'medae': '',
}


def _init_worker(seed):
    np.random.seed(seed)
    random.seed(seed)


def _eval_train_candidate(args):
    probe_set, cand_j, metric, target_by_model, base_matrix = args
    predictions, metrics, score = evaluate_probe_set(
        probe_set,
        predict_benchpress_scores,
        metric=metric,
        target_by_model=target_by_model,
        base_matrix=base_matrix,
    )
    return cand_j, predictions, metrics, score


def _cache_root(out_path, metric, candidate_source, candidate_allowlist_ids,
                train_model_ids, validation_model_ids, train_fraction, seed):
    stem = os.path.basename(out_path)
    if stem.endswith('.json.gz'):
        stem = stem[:-len('.json.gz')]
    else:
        stem = os.path.splitext(stem)[0]
    split_digest = short_text_hash(
        '\n'.join(train_model_ids) + '\n---\n' + '\n'.join(validation_model_ids),
        n=12,
    )
    name = (
        f"{safe_token(stem)}__metric-{safe_token(metric)}"
        f"__candidates-{safe_token(candidate_source)}"
        f"__protocol-{safe_token(MODEL_SPLIT_PROTOCOL)}"
        f"__trainfrac-{safe_token(train_fraction)}"
        f"__seed-{safe_token(seed)}"
        f"__split-{split_digest}"
    )
    if candidate_allowlist_ids is not None:
        digest = short_text_hash('\n'.join(candidate_allowlist_ids), n=12)
        name += f"__allowlist-{digest}"
    return os.path.join(RESULTS_DIR, '.candidate_cache', name)


def _metric_payload(metrics):
    return {
        'medape': float(metrics['medape']) if np.isfinite(metrics['medape']) else None,
        'medae': float(metrics['medae']) if np.isfinite(metrics['medae']) else None,
        'n': int(metrics['n']),
    }


def _candidate_record(cand_j, predictions, metrics, score):
    record = {
        'benchmark_id': BENCH_IDS[cand_j],
        'score': float(score) if np.isfinite(score) else float('inf'),
        'metrics': _metric_payload(metrics),
        'predictions': pack_probe_predictions(predictions),
    }
    return record


def _load_candidate_cache(path, expected_benchmark_id, expected_probe_set):
    if not os.path.exists(path):
        return None
    payload = load_json(path)
    record = payload.get('record', {})
    protocol = payload.get('eval_protocol')
    if protocol != MODEL_SPLIT_PROTOCOL:
        raise RuntimeError(
            f"Candidate cache protocol mismatch in {path}: expected "
            f"{MODEL_SPLIT_PROTOCOL}, found {protocol}. Delete the step cache "
            "or use a different output file."
        )
    if record.get('benchmark_id') != expected_benchmark_id:
        raise RuntimeError(
            f"Candidate cache mismatch in {path}: expected "
            f"{expected_benchmark_id}, found {record.get('benchmark_id')}"
        )
    probe_set = payload.get('probe_set_before_candidate')
    if probe_set != expected_probe_set:
        raise RuntimeError(
            f"Candidate cache mismatch in {path}: expected probe set "
            f"{expected_probe_set}, found {probe_set}. Delete the step cache "
            "or use a different output file."
        )
    return record


def _validate_probe_set(probe_set, metric, train_model_indices, validation_model_indices):
    non_probe_predictions, non_probe_metrics, non_probe_score = (
        evaluate_probe_set_on_heldout_models(
            probe_set,
            predict_benchpress_scores,
            validation_model_indices,
            train_model_indices,
            metric=metric,
            include_probe_targets=False,
        )
    )
    with_probe_predictions, with_probe_metrics, with_probe_score = (
        evaluate_probe_set_on_heldout_models(
            probe_set,
            predict_benchpress_scores,
            validation_model_indices,
            train_model_indices,
            metric=metric,
            include_probe_targets=True,
        )
    )
    return {
        'non_probe': {
            'score': float(non_probe_score) if np.isfinite(non_probe_score) else float('inf'),
            'metrics': _metric_payload(non_probe_metrics),
            'predictions': pack_probe_predictions(non_probe_predictions),
        },
        'with_probe_zero': {
            'score': float(with_probe_score) if np.isfinite(with_probe_score) else float('inf'),
            'metrics': _metric_payload(with_probe_metrics),
            'predictions': pack_probe_predictions(with_probe_predictions),
        },
    }


def _model_id_payload(indices):
    return [MODEL_IDS[int(i)] for i in indices]


def _model_name_payload(indices):
    return {
        MODEL_IDS[int(i)]: MODEL_NAMES.get(MODEL_IDS[int(i)], MODEL_IDS[int(i)])
        for i in indices
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--metric', type=str, default='medae', choices=['medape', 'medae'])
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--candidate-limit', type=int, default=None,
                        help='Debug/smoke-test only: keep the first N candidates.')
    parser.add_argument('--candidate-allowlist', type=str, default=None)
    parser.add_argument('--train-fraction', type=float, default=TRAIN_FRACTION)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--model-limit', type=int, default=None,
                        help='Debug/smoke-test only: split only the first N model rows.')
    args = parser.parse_args()

    candidate_allowlist, candidate_allowlist_ids = load_benchmark_allowlist(
        args.candidate_allowlist, label='Candidate allowlist',
    )
    candidate_source = 'allowlist' if candidate_allowlist is not None else 'all'
    if args.out is None:
        suffix = 'usercheap' if candidate_allowlist is not None else 'all'
        args.out = (
            f"model_split_validation_{args.metric}_"
            f"train{int(round(args.train_fraction * 100)):02d}_{suffix}.json.gz"
        )

    split_model_indices = None
    if args.model_limit is not None:
        split_model_indices = list(range(min(N_MODELS, int(args.model_limit))))
    train_model_indices, validation_model_indices = split_models_for_probe_validation(
        train_fraction=args.train_fraction,
        seed=args.seed,
        model_indices=split_model_indices,
    )
    train_model_ids = _model_id_payload(train_model_indices)
    validation_model_ids = _model_id_payload(validation_model_indices)

    if candidate_allowlist is not None:
        candidates = [BENCH_IDS.index(bid) for bid in candidate_allowlist_ids]
        if not candidates:
            raise RuntimeError("Candidate allowlist produced zero candidate benchmarks")
    else:
        candidates = list(range(N_BENCH))
    if args.candidate_limit is not None:
        candidates = candidates[:args.candidate_limit]

    train_target_by_model = target_by_model_from_models(train_model_indices)
    train_base_matrix = base_matrix_for_model_context(train_model_indices)
    n_train_targets = sum(len(x) for x in train_target_by_model)
    n_validation_targets = int(sum(OBSERVED[i].sum() for i in validation_model_indices))

    print(f"Matrix: {N_MODELS}x{N_BENCH}, observed={int(OBSERVED.sum())}")
    print(
        f"Split: train={len(train_model_indices)} models/{n_train_targets} cells, "
        f"validation={len(validation_model_indices)} models/{n_validation_targets} cells"
    )
    print(f"Candidate set: {candidate_source} ({len(candidates)} benchmarks)")
    print(f"Metric: {args.metric}; workers={args.workers}")

    out_path = os.path.join(RESULTS_DIR, args.out)
    selected = []
    remaining = list(candidates)
    trajectory = []

    expected_resume = {
        'protocol': MODEL_SPLIT_PROTOCOL,
        'metric': args.metric,
        'candidate_source': candidate_source,
        'candidate_allowlist_path': (
            os.path.relpath(args.candidate_allowlist, REPO_ROOT)
            if args.candidate_allowlist else None
        ),
        'candidate_allowlist_ids': candidate_allowlist_ids,
        'candidate_limit': args.candidate_limit,
        'train_fraction': args.train_fraction,
        'seed': args.seed,
        'model_limit': args.model_limit,
        'train_model_ids': train_model_ids,
        'validation_model_ids': validation_model_ids,
        'n_candidates': len(candidates),
    }
    if os.path.exists(out_path):
        prev = load_json(out_path)
        prev_config = prev.get('config', {})
        prev_resume = {k: prev_config.get(k) for k in expected_resume}
        if prev_resume != expected_resume:
            raise SystemExit(
                f"Refusing to resume {out_path}: existing config {prev_resume} "
                f"does not match requested config {expected_resume}. Pick a "
                f"different --out or delete the file."
            )
        if prev.get('trajectory'):
            print(f"\nResuming from {out_path} "
                  f"({len(prev['trajectory'])} steps already done).")
            trajectory = prev['trajectory']
            selected = [BENCH_IDS.index(s['added_benchmark']) for s in trajectory]
            remaining = [j for j in candidates if j not in selected]

    cache_root = _cache_root(
        out_path,
        args.metric,
        candidate_source,
        candidate_allowlist_ids,
        train_model_ids,
        validation_model_ids,
        args.train_fraction,
        args.seed,
    )
    t_all = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.seed,),
    ) as pool:
        for step in range(len(selected) + 1, args.max_steps + 1):
            if not remaining:
                break
            print(f"\n--- Step {step}/{args.max_steps} ({len(remaining)} candidates) ---")
            step_t0 = time.time()

            selected_ids = [BENCH_IDS[j] for j in selected]
            candidate_results = {}
            futures = {}
            for cand_j in remaining:
                cache_path = probe_candidate_cache_path(cache_root, step, BENCH_IDS[cand_j])
                cached = _load_candidate_cache(cache_path, BENCH_IDS[cand_j], selected_ids)
                if cached is not None:
                    candidate_results[BENCH_IDS[cand_j]] = cached
                    continue
                future = pool.submit(
                    _eval_train_candidate,
                    (
                        selected + [cand_j],
                        cand_j,
                        args.metric,
                        train_target_by_model,
                        train_base_matrix,
                    ),
                )
                futures[future] = cand_j

            for future in as_completed(futures):
                cand_j, predictions, metrics, score = future.result()
                record = _candidate_record(cand_j, predictions, metrics, score)
                write_json_atomic(
                    probe_candidate_cache_path(cache_root, step, BENCH_IDS[cand_j]),
                    {
                        'step': step,
                        'probe_set_before_candidate': selected_ids,
                        'metric': args.metric,
                        'candidate_source': candidate_source,
                        'eval_protocol': MODEL_SPLIT_PROTOCOL,
                        'record': record,
                    },
                )
                candidate_results[BENCH_IDS[cand_j]] = record

            if len(candidate_results) != len(remaining):
                raise RuntimeError(
                    f"Step {step} has {len(candidate_results)} candidate results, "
                    f"expected {len(remaining)}"
                )

            best_score = float('inf')
            best_j = None
            best_record = None
            for cand_j in remaining:
                record = candidate_results[BENCH_IDS[cand_j]]
                score = record['score']
                if score < best_score:
                    best_score = score
                    best_j = cand_j
                    best_record = record
            selected.append(best_j)
            remaining.remove(best_j)

            validation = _validate_probe_set(
                selected,
                args.metric,
                train_model_indices,
                validation_model_indices,
            )
            step_record = {
                'step': step,
                'added_benchmark': BENCH_IDS[best_j],
                'added_benchmark_name': BENCH_NAMES.get(BENCH_IDS[best_j], BENCH_IDS[best_j]),
                'probe_set': [BENCH_IDS[j] for j in selected],
                'train': {
                    'score': best_score,
                    'metrics': best_record['metrics'],
                },
                'validation_non_probe': validation['non_probe'],
                'validation_with_probe_zero': validation['with_probe_zero'],
                'elapsed_s': time.time() - step_t0,
                'candidate_results': candidate_results,
            }
            trajectory.append(step_record)
            unit = SCORE_UNIT[args.metric]
            val_score = validation['non_probe']['score']
            print(
                f"  -> Added {BENCH_IDS[best_j]:30s} "
                f"train={best_score:.3f}{unit} val_non_probe={val_score:.3f}{unit} "
                f"[{time.time() - step_t0:.1f}s]"
            )

            output = {
                'config': {
                    **expected_resume,
                    'n_models': N_MODELS,
                    'n_bench': N_BENCH,
                    'n_observed': int(OBSERVED.sum()),
                    'n_train_models': len(train_model_indices),
                    'n_validation_models': len(validation_model_indices),
                    'n_train_target_cells': n_train_targets,
                    'n_validation_observed_cells': n_validation_targets,
                    'bench_ids': BENCH_IDS,
                    'train_model_names': _model_name_payload(train_model_indices),
                    'validation_model_names': _model_name_payload(validation_model_indices),
                    'validation_eval_scope': (
                        'held-out model rows; non-probe metrics exclude already '
                        'measured probe cells, with_probe_zero metrics include '
                        'observed probe cells as pred=true for compatibility with '
                        'the all-known-cell probe denominator'
                    ),
                    'cell_masking': (
                        'Probe selection uses only training model rows. Validation '
                        'isolates each held-out target model: the predictor sees '
                        'training rows plus that target model probe cells, and '
                        'does not see other held-out model rows.'
                    ),
                    'prediction_engine': (
                        'predict_benchpress_scores (Logit Bias ALS, rank=2, lambda=0.1)'
                    ),
                    'workers': args.workers,
                    'candidate_cache_dir': os.path.relpath(cache_root, SCRIPT_DIR),
                },
                'split': {
                    'train_model_ids': train_model_ids,
                    'validation_model_ids': validation_model_ids,
                },
                'trajectory': trajectory,
            }
            write_json_atomic(out_path, output, indent=2)

    print(f"\nSaved -> {out_path}")
    print(f"Total time: {time.time() - t_all:.1f}s")


if __name__ == '__main__':
    main()
