#!/usr/bin/env python3
"""
Hero figure raw-prediction experiment.

This script is the only compute entry point for Figure 1:
  - default: run the full serial sweep and write results.json
  - --k K: run all seeds for one k in parallel
  - --k K --seed S --output PATH: run one (k, seed) shard
  - --merge --shard-dir DIR: merge shards into results.json
"""

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from benchpress.all_methods import predict_benchpress_scores
from benchpress.evaluation_harness import (
    BENCH_IDS,
    M_FULL,
    MODEL_IDS,
    N_BENCH,
    N_MODELS,
    print_matrix_summary,
    random_model_keep_k_split,
)
from benchpress.shard_utils import (
    default_k_seed_shard_name,
    merge_prediction_shards,
    run_k_seed_shards,
    validate_k_seed_result_payload,
    write_prediction_result,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")
DEFAULT_SHARD_DIR = os.path.join(SCRIPT_DIR, "shards")

K_MAX = 20
N_SEEDS = 10
BASE_SEED = 42

np.random.seed(BASE_SEED)


def run_one(k, seed_idx):
    """Run one (k, seed) combo and return per-cell raw predictions."""
    raw = []
    n_calls = 0
    start = time.time()

    for i in range(N_MODELS):
        _, masked = random_model_keep_k_split(i, k, seed_idx, base_seed=BASE_SEED)
        if not masked:
            continue

        m_train = M_FULL.copy()
        for j in masked:
            m_train[i, j] = np.nan

        m_pred = predict_benchpress_scores(m_train)
        n_calls += 1

        for j in masked:
            actual = float(M_FULL[i, j])
            pred = float(m_pred[i, j])
            if np.isfinite(actual) and np.isfinite(pred):
                raw.append(
                    {
                        "seed": seed_idx,
                        "k": k,
                        "model": i,
                        "bench": j,
                        "actual": round(actual, 6),
                        "pred": round(pred, 6),
                    }
                )

    return raw, n_calls, time.time() - start


def _result_config(k_max=K_MAX, n_seeds=N_SEEDS):
    return {
        "k_max": k_max,
        "n_seeds": n_seeds,
        "base_seed": BASE_SEED,
        "n_models": N_MODELS,
        "n_bench": N_BENCH,
        "model_ids": MODEL_IDS,
        "bench_ids": BENCH_IDS,
    }


def write_hero_result(raw, output_path, k_max=K_MAX, n_seeds=N_SEEDS):
    write_prediction_result(raw, output_path, _result_config(k_max, n_seeds))


def write_single_shard(k, seed_idx, output_path):
    print(f"Single shard: k={k} seed={seed_idx} -> {output_path}", flush=True)
    raw, n_calls, dt = run_one(k, seed_idx)
    print(
        f"k={k} seed={seed_idx}: {n_calls} predict calls, "
        f"{len(raw)} raw rows, {dt:.0f}s",
        flush=True,
    )
    write_hero_result(raw, output_path, k_max=k, n_seeds=1)


def _expected_shard_name(k, seed_idx):
    return default_k_seed_shard_name(k, seed_idx)


def _validate_shard_payload(data, path, expected_k, expected_seed):
    validate_k_seed_result_payload(
        data,
        path,
        expected_k,
        expected_seed,
        expected_config=_result_config(k_max=expected_k, n_seeds=1),
    )


def run_hero_shards(k_values, shard_dir, workers):
    run_k_seed_shards(
        k_values,
        N_SEEDS,
        shard_dir,
        workers,
        shard_name_fn=_expected_shard_name,
        write_shard_fn=write_single_shard,
        validate_shard_fn=_validate_shard_payload,
    )


def merge_hero_shards(shard_dir, output_path):
    expected_names = {
        _expected_shard_name(k, seed_idx)
        for k in range(1, K_MAX + 1)
        for seed_idx in range(N_SEEDS)
    }
    raw = merge_prediction_shards(shard_dir, expected_names, _validate_shard_payload)
    write_hero_result(raw, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="k value (1..K_MAX)")
    parser.add_argument("--seed", type=int, default=None, help="seed index (0..N_SEEDS-1)")
    parser.add_argument("--output", type=str, default=RESULTS_PATH, help="output JSON path")
    parser.add_argument("--shard-dir", type=str, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--merge", action="store_true", help="merge shard-dir into output")
    args = parser.parse_args()

    if args.merge:
        merge_hero_shards(args.shard_dir, args.output)
    elif args.k is not None and args.seed is not None:
        write_single_shard(args.k, args.seed, args.output)
    elif args.k is not None:
        run_hero_shards([args.k], args.shard_dir, args.workers)
    else:
        print_matrix_summary()
        print(f"K_MAX={K_MAX}, N_SEEDS={N_SEEDS}\n")
        run_hero_shards(list(range(1, K_MAX + 1)), args.shard_dir, args.workers)
        merge_hero_shards(args.shard_dir, args.output)


if __name__ == "__main__":
    main()
