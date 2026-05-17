#!/usr/bin/env python3
"""H6 ablation: nested masks for strongest-neighbor support.

For each target benchmark and seed, we fix one hide-half split and one
permutation of the target/neighbor overlap rows. The 25%, 50%, and 75% masks
are nested prefixes of that same permutation, so the only changing factor is
how much best-neighbor evidence is removed.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from benchpress.io_utils import append_jsonl, load_jsonl_keyed, write_json
from benchpress.stats import wilcoxon_per_benchmark

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _shared import (  # noqa: E402
    SEED, M_FULL, N_BENCH, OBSERVED, BENCH_IDS,
    predict_benchpress_scores, compute_prediction_error,
    holdout_half_per_benchmark, mask_cells, pairwise_benchmark_corr,
)

N_SEEDS = 5
DROP_RATES = [0.25, 0.50, 0.75]
METRICS = ["medape", "medae"]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_PATH = os.path.join(HERE, "ablation_records_nested.jsonl")
DEFAULT_OUTPUT_PATH = os.path.join(HERE, "ablation_results.json")


def _record_key(record):
    return (record["bench_id"], int(record["seed"]), float(record["drop_rate"]))


def _load_cache(path):
    return load_jsonl_keyed(path, _record_key, missing_ok=True)


def _append_cache(path, record):
    append_jsonl(path, record, sort_keys=True)


def _raw_predictions(test_idx, bench_idx, true_scores, pred, valid_mask):
    return [
        [
            int(test_idx[i]),
            int(bench_idx),
            round(float(true_scores[i]), 6),
            round(float(pred[i]), 6),
        ]
        for i in range(len(test_idx)) if valid_mask[i]
    ]


def _best_neighbors():
    corr, _ = pairwise_benchmark_corr()
    neighbors = []
    for j in range(N_BENCH):
        row = corr[j].copy()
        row[j] = np.nan
        if np.any(np.isfinite(row)):
            neighbors.append(int(np.nanargmax(row)))
        else:
            neighbors.append(None)
    return neighbors


def _eligible_benchmark_indices(neighbors, max_benchmarks=None):
    indices = []
    for j in range(N_BENCH):
        obs_rows = np.where(OBSERVED[:, j])[0]
        if len(obs_rows) < 6:
            continue
        neighbor = neighbors[j]
        if neighbor is None:
            continue
        overlap_rows = np.where(OBSERVED[:, j] & OBSERVED[:, neighbor])[0]
        if len(overlap_rows) < 4:
            continue
        indices.append(j)
        if max_benchmarks is not None and len(indices) >= max_benchmarks:
            break
    return indices


def _run_one(j, neighbor, seed_idx, cache, cache_path):
    overlap_rows = np.where(OBSERVED[:, j] & OBSERVED[:, neighbor])[0]

    split_rng = np.random.RandomState(SEED + 10_000 + 1_000 * seed_idx + j)
    mask_rng = np.random.RandomState(SEED + 20_000 + 1_000 * seed_idx + j)
    test_idx, _ = holdout_half_per_benchmark(j, split_rng, min_test=3)
    true_scores = M_FULL[test_idx, j]
    overlap_perm = overlap_rows[mask_rng.permutation(len(overlap_rows))]

    M_base = mask_cells((idx, j) for idx in test_idx)
    P_base = predict_benchpress_scores(M_base)
    pred_base = P_base[test_idx, j]
    valid_base = np.isfinite(pred_base)
    if valid_base.sum() < 2:
        return 0

    n_written = 0
    for drop_rate in DROP_RATES:
        key = (BENCH_IDS[j], int(seed_idx), float(drop_rate))
        if key in cache:
            continue

        n_drop = int(round(len(overlap_perm) * drop_rate))
        dropped_rows = overlap_perm[:n_drop]
        M_masked = mask_cells(((idx, neighbor) for idx in dropped_rows), base_matrix=M_base)
        P_treat = predict_benchpress_scores(M_masked)
        pred_treat = P_treat[test_idx, j]
        valid_treat = np.isfinite(pred_treat)
        paired = valid_base & valid_treat
        if paired.sum() < 2:
            continue

        base_metrics = compute_prediction_error(true_scores[paired], pred_base[paired])
        treat_metrics = compute_prediction_error(true_scores[paired], pred_treat[paired])

        record = {
            "bench_id": BENCH_IDS[j],
            "best_neighbor": BENCH_IDS[neighbor],
            "seed": int(seed_idx),
            "drop_rate": float(drop_rate),
            "mask_strategy": "nested_overlap_prefix",
            "n_overlap_original": int(len(overlap_rows)),
            "n_neighbor_original": int(np.isfinite(M_FULL[:, neighbor]).sum()),
            "n_neighbor_kept_in_overlap": int(len(overlap_rows) - n_drop),
            "n_test": int(paired.sum()),
            "dropped_model_indices": [int(i) for i in dropped_rows],
            "raw_base": _raw_predictions(test_idx, j, true_scores, pred_base, paired),
            "raw_treat": _raw_predictions(test_idx, j, true_scores, pred_treat, paired),
        }
        for metric in METRICS:
            record[f"base_{metric}"] = float(base_metrics[metric])
            record[f"treat_{metric}"] = float(treat_metrics[metric])
            record[f"delta_{metric}"] = (
                float(treat_metrics[metric]) - float(base_metrics[metric])
            )

        cache[key] = record
        _append_cache(cache_path, record)
        n_written += 1
    return n_written


def _read_cache_paths(paths):
    records = {}
    for path in paths:
        records.update(_load_cache(path))
    return records


def _make_results(records, metadata):
    selected_records = list(records.values())
    selected_records.sort(
        key=lambda r: (r["bench_id"], int(r["seed"]), float(r["drop_rate"]))
    )
    out = {
        "records": selected_records,
        "wilcoxon": wilcoxon_per_benchmark(
            [r for r in selected_records if float(r["drop_rate"]) == 0.75],
            METRICS,
            invalid_median=0.0,
            invalid_p=1.0,
        ),
        "hypothesis": "H7",
        "intervention": "nested_neighbor_overlap_mask",
        "drop_rates": DROP_RATES,
        "n_seeds": int(metadata["max_seeds"]),
        "mask_strategy": "nested_overlap_prefix",
        **metadata,
    }
    return out


def _write_results(records, output_path, metadata):
    out = _make_results(records, metadata)
    write_json(output_path, out, indent=2)
    return out


def ablation_h7_neighbor_support(max_benchmarks=None, max_seeds=N_SEEDS,
                                 cache_path=DEFAULT_CACHE_PATH,
                                 shard_index=0, num_shards=1):
    records = _load_cache(cache_path)
    t0 = time.time()
    neighbors = _best_neighbors()
    bench_indices = _eligible_benchmark_indices(
        neighbors, max_benchmarks=max_benchmarks
    )
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    bench_indices = [
        j for pos, j in enumerate(bench_indices)
        if pos % num_shards == shard_index
    ]
    for pos, j in enumerate(bench_indices, start=1):
        for seed_idx in range(max_seeds):
            _run_one(j, neighbors[j], seed_idx, records, cache_path)
        print(
            f"  H7 nested [{pos:2d}/{len(bench_indices)}] "
            f"{BENCH_IDS[j]:30s} ({time.time() - t0:.0f}s)"
        )

    selected_bench_ids = {BENCH_IDS[j] for j in bench_indices}
    return _make_results(
        {
            k: r for k, r in records.items()
            if r["bench_id"] in selected_bench_ids and int(r["seed"]) < max_seeds
        },
        metadata={
            "max_seeds": int(max_seeds),
            "cache_path": cache_path,
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "n_benchmarks": int(len(bench_indices)),
        },
    )


def merge_shards(cache_glob, output_path, max_seeds=N_SEEDS):
    paths = sorted(glob.glob(cache_glob))
    if not paths:
        raise FileNotFoundError(f"No cache files matched: {cache_glob}")
    records = _read_cache_paths(paths)
    records = {
        k: r for k, r in records.items()
        if int(r["seed"]) < max_seeds
    }
    return _write_results(
        records,
        output_path,
        metadata={
            "max_seeds": int(max_seeds),
            "cache_glob": cache_glob,
            "cache_files": paths,
            "n_cache_files": int(len(paths)),
            "n_benchmarks": int(len({r["bench_id"] for r in records.values()})),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-benchmarks", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-cache-glob", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)
    if args.merge_cache_glob:
        out = merge_shards(
            args.merge_cache_glob,
            args.output_path,
            max_seeds=args.max_seeds,
        )
        print(f"[saved] {args.output_path} ({len(out['records'])} records)")
        print("\n== Wilcoxon summary ==")
        for metric, value in out["wilcoxon"].items():
            print(
                f"    {metric:10s}  Δmedian={value['median_delta']:+.3f}  "
                f"p={value['p_value']:.4f}  n={value['n']}"
            )
        return

    if args.force:
        for path in [args.cache_path, args.output_path]:
            if os.path.exists(path):
                os.remove(path)
    out = ablation_h7_neighbor_support(
        max_benchmarks=args.max_benchmarks,
        max_seeds=args.max_seeds,
        cache_path=args.cache_path,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    write_json(args.output_path, out, indent=2)
    print(f"[saved] {args.output_path} ({len(out['records'])} records)")
    print("\n== Wilcoxon summary ==")
    for metric, value in out["wilcoxon"].items():
        print(
            f"    {metric:10s}  Δmedian={value['median_delta']:+.3f}  "
            f"p={value['p_value']:.4f}  n={value['n']}"
        )


if __name__ == "__main__":
    main()
