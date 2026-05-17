#!/usr/bin/env python3
"""H6 ablation: nested strongest-peer support.

For each target model and seed, hide half of the target model's observed
benchmarks, identify the strongest peer model (highest |r|, requiring
|r| >= NEIGHBOR_THRESH), and mask nested fractions of the target/peer shared
benchmarks in that peer row. The 25%, 50%, and 75% masks are prefixes of the
same permutation, mirroring benchmark-side H7.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from benchpress.stats import wilcoxon_grouped_median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _shared import (  # noqa: E402
    SEED, N_SEEDS, OBSERVED, NEIGHBOR_THRESH,
    M_FULL, N_MODELS, predict_benchpress_scores,
    pairwise_abs_r, compute_prediction_error, MODEL_IDS,
)
from benchpress.io_utils import write_json  # noqa: E402
from benchpress.shard_utils import (  # noqa: E402
    collect_unit_records,
    load_json_unit_shards,
    unit_shard_path,
    write_unit_json_atomic,
)

DROP_RATES = [0.25, 0.50, 0.75]
METRICS = ["medape", "medae"]
HERE = os.path.dirname(os.path.abspath(__file__))
SHARDS_DIR = os.path.join(HERE, "shards")
RESULTS_PATH = os.path.join(HERE, "results.json")


def _raw_predictions(test_idx, model_idx, true_scores, pred, valid_mask):
    return [
        [
            int(model_idx),
            int(test_idx[j]),
            round(float(true_scores[j]), 6),
            round(float(pred[j]), 6),
        ]
        for j in range(len(test_idx)) if valid_mask[j]
    ]


def _strongest_peers(R):
    peers = []
    for i in range(N_MODELS):
        row = R[i].copy()
        row[i] = np.nan
        eligible = np.where(row >= NEIGHBOR_THRESH)[0]
        if eligible.size == 0:
            peers.append(None)
            continue
        best = eligible[np.nanargmax(row[eligible])]
        peers.append(int(best))
    return peers


def _precompute_units(max_models=None, max_seeds=N_SEEDS):
    R = pairwise_abs_r()
    strongest_peers = _strongest_peers(R)
    units = []
    unit_index = 0

    for i in range(N_MODELS):
        obs_j = np.where(OBSERVED[i])[0]
        if len(obs_j) < 4:
            continue
        peer = strongest_peers[i]
        if peer is None:
            continue
        overlap = np.where(OBSERVED[i] & OBSERVED[peer])[0]
        if len(overlap) == 0:
            continue
        for seed in range(max_seeds):
            split_rng = np.random.RandomState(SEED + 10_000 + seed * 1_000 + i)
            mask_rng = np.random.RandomState(SEED + 20_000 + seed * 1_000 + i)
            obs_perm = obs_j.copy()
            split_rng.shuffle(obs_perm)
            test_idx = obs_perm[:len(obs_perm) // 2]
            overlap_perm = overlap[mask_rng.permutation(len(overlap))]
            units.append({
                "unit_index": int(unit_index),
                "model_index": int(i),
                "model_id": MODEL_IDS[i],
                "seed": int(seed),
                "peer_index": int(peer),
                "peer_id": MODEL_IDS[peer],
                "test_idx": test_idx.astype(int).tolist(),
                "overlap_perm": overlap_perm.astype(int).tolist(),
                "n_obs_original": int(len(obs_j)),
                "n_overlap_original": int(len(overlap)),
                "peer_abs_r": float(R[i, peer]),
            })
            unit_index += 1
        if max_models is not None and len({u["model_index"] for u in units}) >= max_models:
            break
    return units


def _run_unit(unit):
    i = int(unit["model_index"])
    peer = int(unit["peer_index"])
    test_idx = np.array(unit["test_idx"], dtype=int)
    overlap_perm = np.array(unit["overlap_perm"], dtype=int)
    true_scores = M_FULL[i, test_idx]

    M_base = M_FULL.copy()
    M_base[i, test_idx] = np.nan
    P_base = predict_benchpress_scores(M_base)
    pred_base = P_base[i, test_idx]
    valid_base = np.isfinite(pred_base)
    if valid_base.sum() < 2:
        return {**unit, "records": []}

    records = []
    for drop_rate in DROP_RATES:
        n_drop = max(1, int(round(len(overlap_perm) * drop_rate)))
        dropped = overlap_perm[:n_drop]
        M_treat = M_base.copy()
        M_treat[peer, dropped] = np.nan
        P_treat = predict_benchpress_scores(M_treat)
        pred_treat = P_treat[i, test_idx]
        paired = valid_base & np.isfinite(pred_treat)
        if paired.sum() < 2:
            continue

        base_metrics = compute_prediction_error(true_scores[paired], pred_base[paired])
        treat_metrics = compute_prediction_error(true_scores[paired], pred_treat[paired])
        record = {
            "model_id": unit["model_id"],
            "model_index": i,
            "seed": int(unit["seed"]),
            "strongest_peer": unit["peer_id"],
            "strongest_peer_index": peer,
            "peer_abs_r": unit["peer_abs_r"],
            "drop_rate": float(drop_rate),
            "mask_strategy": "nested_overlap_prefix",
            "n_obs_original": int(unit["n_obs_original"]),
            "n_overlap_original": int(unit["n_overlap_original"]),
            "n_peer_kept_in_overlap": int(len(overlap_perm) - n_drop),
            "n_test": int(paired.sum()),
            "dropped_benchmark_indices": [int(j) for j in dropped],
            "raw_base": _raw_predictions(test_idx, i, true_scores, pred_base, paired),
            "raw_treat": _raw_predictions(test_idx, i, true_scores, pred_treat, paired),
        }
        for metric in METRICS:
            record[f"base_{metric}"] = float(base_metrics[metric])
            record[f"treat_{metric}"] = float(treat_metrics[metric])
            record[f"delta_{metric}"] = (
                float(treat_metrics[metric]) - float(base_metrics[metric])
            )
        records.append(record)
    return {**unit, "records": records}


def summarize_records(records):
    by_drop = {}
    for drop_rate in DROP_RATES:
        drop_records = [r for r in records if float(r["drop_rate"]) == drop_rate]
        by_drop[str(drop_rate)] = wilcoxon_grouped_median(
            drop_records,
            METRICS,
            group_key="model_id",
            p_key="p",
            include_sign_counts=True,
        )
    return by_drop


def merge_h6_shards(shards_dir=SHARDS_DIR, output_path=RESULTS_PATH):
    units = load_json_unit_shards(shards_dir, label="H6 shard JSON files")
    if not units:
        raise FileNotFoundError(f"No H6 shard JSON files found in {shards_dir}")
    records = collect_unit_records(units)
    by_drop_rate = summarize_records(records)
    out = {
        "hypothesis": "H6_neighbor_evidence",
        "intervention": "nested_strongest_peer_overlap_mask",
        "neighbor_thresh": NEIGHBOR_THRESH,
        "drop_rates": DROP_RATES,
        "headline_drop_rate": 0.75,
        "mask_strategy": "nested_overlap_prefix",
        "n_seeds": N_SEEDS,
        "n_units": len(units),
        "records": records,
        "by_drop_rate": by_drop_rate,
        "tests": by_drop_rate["0.75"],
    }
    write_json(output_path, out, indent=2)
    return out


def run_h6(max_models=None, max_seeds=N_SEEDS, workers=1, num_shards=1,
           shard_id=0, shards_dir=SHARDS_DIR, force=False):
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")

    os.makedirs(shards_dir, exist_ok=True)
    units = _precompute_units(max_models=max_models, max_seeds=max_seeds)
    units = [u for u in units if int(u["unit_index"]) % num_shards == shard_id]
    if force:
        for unit in units:
            path = unit_shard_path(shards_dir, "model", unit["model_id"], int(unit["seed"]))
            if os.path.exists(path):
                os.remove(path)
    pending = [
        u for u in units
        if not os.path.exists(unit_shard_path(shards_dir, "model", u["model_id"], int(u["seed"])))
    ]

    print(
        f"[H6] shard {shard_id}/{num_shards}: {len(units)} units, "
        f"{len(pending)} pending, workers={workers}",
        flush=True,
    )
    t0 = time.time()

    if workers == 1:
        for done, unit in enumerate(pending, start=1):
            unit_out = _run_unit(unit)
            write_unit_json_atomic(
                shards_dir, "model", unit_out["model_id"], int(unit_out["seed"]), unit_out
            )
            print(
                f"  H6 unit {done:4d}/{len(pending)} "
                f"{unit['model_id']} seed={unit['seed']} ({time.time()-t0:.0f}s)",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_unit, unit): unit for unit in pending}
            for done, fut in enumerate(as_completed(futures), start=1):
                unit = futures[fut]
                unit_out = fut.result()
                write_unit_json_atomic(
                    shards_dir, "model", unit_out["model_id"], int(unit_out["seed"]), unit_out
                )
                print(
                    f"  H6 unit {done:4d}/{len(pending)} "
                    f"{unit['model_id']} seed={unit['seed']} ({time.time()-t0:.0f}s)",
                    flush=True,
                )

    if num_shards == 1:
        return merge_h6_shards(shards_dir=shards_dir)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shards-dir", default=SHARDS_DIR)
    parser.add_argument("--output-path", default=RESULTS_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    np.random.seed(SEED)
    t0 = time.time()
    if args.merge_only:
        out = merge_h6_shards(shards_dir=args.shards_dir, output_path=args.output_path)
    else:
        out = run_h6(
            max_models=args.max_models,
            max_seeds=args.max_seeds,
            workers=args.workers,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            shards_dir=args.shards_dir,
            force=args.force,
        )
        if out is None:
            print("[H6] shard complete; run --merge-only after all shards finish")
            return
        if args.output_path != RESULTS_PATH:
            write_json(args.output_path, out, indent=2)

    print(f"[H6] Done ({time.time()-t0:.0f}s)")
    print(f"[saved] {args.output_path} ({len(out['records'])} records)")
    print("\n== Headline Wilcoxon summary (drop_rate=0.75) ==")
    for metric, v in out["tests"].items():
        print(
            f"    {metric:10s}  Δmedian={v['median_delta']:+.3f}  "
            f"p={v['p']:.4f}  n={v['n']}"
        )


if __name__ == "__main__":
    main()
