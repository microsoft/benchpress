#!/usr/bin/env python3
"""Random nested probe-prefix baseline using the Figure 1 all-known-cell protocol.

For each seed, draw one global random ordering of benchmark columns. For each k,
use the first k columns from that ordering. Every model uses that same probe
prefix, matching the greedy probe-set protocol. A cell is revealed only when the
target model has an observed score in one of those global probe columns; revealed
cells get pred=true, and all unrevealed known cells are predicted with
BenchPress. Raw per-cell predictions are saved so downstream plots can aggregate
over all cells or any benchmark subset.
"""

import argparse
import os
import random
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from benchpress.all_methods import predict_benchpress_scores
from benchpress.evaluation_harness import (
    BENCH_IDS,
    MODEL_IDS,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    evaluate_probe_set,
    observed_benchmarks_by_model,
    random_global_probe_set,
)
from benchpress.shard_utils import (
    default_k_seed_shard_name,
    merge_prediction_shards,
    run_k_seed_shards,
    validate_k_seed_result_payload,
    write_prediction_result,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "random_medape_hero_all_known.json.gz")
DEFAULT_SHARD_DIR = os.path.join(RESULTS_DIR, "random_medape_hero_all_known_nested_shards")

K_MAX = 30
N_SEEDS = 10
BASE_SEED = 42
RANDOM_PROTOCOL = "figure1_random_nested_probe_prefix_all_known_cells"

np.random.seed(BASE_SEED)
random.seed(BASE_SEED)


def _probe_set_for(k, seed_idx):
    return random_global_probe_set(k, seed_idx, base_seed=BASE_SEED)


def run_one(k, seed_idx, model_limit=None):
    """Run one (k, seed) shard and return raw per-cell predictions."""
    probe_set = _probe_set_for(k, seed_idx)
    raw = []
    n_calls = 0
    start = time.time()

    def predict_and_count(m_train):
        nonlocal n_calls
        n_calls += 1
        return predict_benchpress_scores(m_train)

    predictions, _, _ = evaluate_probe_set(
        probe_set,
        predict_and_count,
        metric="medape",
        target_by_model=observed_benchmarks_by_model(model_limit),
    )

    for i, j, actual, pred in sorted(predictions, key=lambda p: (p[0], p[1])):
        if np.isfinite(actual) and np.isfinite(pred):
            raw.append({
                "seed": int(seed_idx),
                "k": int(k),
                "model": int(i),
                "bench": int(j),
                "actual": round(actual, 6),
                "pred": round(pred, 6),
            })

    return raw, n_calls, time.time() - start


def _result_config(k_max=K_MAX, n_seeds=N_SEEDS, model_limit=None):
    return {
        "protocol": RANDOM_PROTOCOL,
        "k_max": k_max,
        "n_seeds": n_seeds,
        "base_seed": BASE_SEED,
        "model_limit": model_limit,
        "n_models": N_MODELS,
        "n_bench": N_BENCH,
        "n_observed": int(OBSERVED.sum()),
        "n_target_cells": int(OBSERVED.sum()) if model_limit is None else None,
        "eval_scope": "all_observed_cells",
        "model_ids": MODEL_IDS,
        "bench_ids": BENCH_IDS,
        "prediction_engine": "predict_benchpress_scores",
        "cell_masking": (
            "For each seed, choose one global random benchmark ordering. "
            "For each k, use the first k columns from that ordering. Every "
            "model uses that same probe prefix; a revealed cell contributes "
            "pred=true only if that model has an observed score in a selected "
            "probe column. Unrevealed known cells are predicted by BenchPress. "
            "The evaluation universe is fixed to all observed cells for every k."
        ),
    }


def write_random_probe_result(raw, output_path, k_max=K_MAX, n_seeds=N_SEEDS, model_limit=None):
    write_prediction_result(
        raw,
        output_path,
        _result_config(k_max=k_max, n_seeds=n_seeds, model_limit=model_limit),
        include_summary_by_k_seed=True,
    )


def _expected_shard_name(k, seed_idx, model_limit):
    suffix = f"_m{model_limit}" if model_limit is not None else ""
    return default_k_seed_shard_name(k, seed_idx, suffix=suffix, extension=".json.gz")


def _validate_shard_payload(data, path, expected_k, expected_seed, model_limit):
    expected_config = {
        "protocol": RANDOM_PROTOCOL,
        "k_max": expected_k,
        "n_seeds": 1,
        "base_seed": BASE_SEED,
        "model_limit": model_limit,
        "n_models": N_MODELS,
        "n_bench": N_BENCH,
    }
    validate_k_seed_result_payload(
        data,
        path,
        expected_k,
        expected_seed,
        expected_config=expected_config,
        expected_summary_n=int(OBSERVED.sum()) if model_limit is None else None,
        require_summary=True,
    )


def write_single_shard(k, seed_idx, output_path, model_limit=None):
    print(f"Single shard: k={k} seed={seed_idx} -> {output_path}", flush=True)
    raw, n_calls, dt = run_one(k, seed_idx, model_limit=model_limit)
    print(
        f"k={k} seed={seed_idx}: {n_calls} predict calls, "
        f"{len(raw)} raw rows, {dt:.0f}s",
        flush=True,
    )
    write_random_probe_result(raw, output_path, k_max=k, n_seeds=1, model_limit=model_limit)


def run_random_probe_shards(k_values, n_seeds, shard_dir, workers, model_limit=None):
    run_k_seed_shards(
        k_values,
        n_seeds,
        shard_dir,
        workers,
        shard_name_fn=lambda k, seed_idx: _expected_shard_name(k, seed_idx, model_limit),
        write_shard_fn=write_single_shard,
        validate_shard_fn=_validate_shard_payload,
        job_kwargs={"model_limit": model_limit},
        label_extra=f"model_limit={model_limit}",
    )


def merge_random_probe_shards(shard_dir, output_path, k_max, n_seeds, model_limit=None):
    expected_names = {
        _expected_shard_name(k, seed_idx, model_limit)
        for k in range(1, k_max + 1)
        for seed_idx in range(n_seeds)
    }
    raw = merge_prediction_shards(
        shard_dir,
        expected_names,
        _validate_shard_payload,
        validation_kwargs={"model_limit": model_limit},
    )
    write_random_probe_result(
        raw,
        output_path,
        k_max=k_max,
        n_seeds=n_seeds,
        model_limit=model_limit,
    )


def _parse_k_values(spec, k_max):
    if spec:
        values = []
        for item in spec.split(","):
            item = item.strip()
            if "-" in item:
                lo, hi = item.split("-", 1)
                values.extend(range(int(lo), int(hi) + 1))
            elif item:
                values.append(int(item))
        return sorted(set(values))
    return list(range(1, k_max + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="single k value")
    parser.add_argument("--seed", type=int, default=None, help="single seed index")
    parser.add_argument("--k-values", type=str, default=None, help="parallel k list/range, e.g. 1-10")
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--output", type=str, default=RESULTS_PATH)
    parser.add_argument("--shard-dir", type=str, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--model-limit", type=int, default=None, help="smoke-test only")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_random_probe_shards(
            args.shard_dir, args.output, args.k_max, args.n_seeds,
            model_limit=args.model_limit,
        )
    elif args.k is not None and args.seed is not None:
        write_single_shard(args.k, args.seed, args.output, model_limit=args.model_limit)
    elif args.k is not None:
        run_random_probe_shards(
            [args.k], args.n_seeds, args.shard_dir, args.workers,
            model_limit=args.model_limit,
        )
    elif args.k_values is not None:
        run_random_probe_shards(
            _parse_k_values(args.k_values, args.k_max), args.n_seeds,
            args.shard_dir, args.workers, model_limit=args.model_limit,
        )
    else:
        run_random_probe_shards(
            _parse_k_values(None, args.k_max), args.n_seeds, args.shard_dir,
            args.workers, model_limit=args.model_limit,
        )
        merge_random_probe_shards(
            args.shard_dir, args.output, args.k_max, args.n_seeds,
            model_limit=args.model_limit,
        )


if __name__ == "__main__":
    main()
