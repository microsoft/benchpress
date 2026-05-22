#!/usr/bin/env python3
"""H4 ablation: drop training observations of the target benchmark.

For each target benchmark j, drop a fraction f of its observed cells from
training and predict the held-out test cells.  Compare against f=0 baseline.
Paired Wilcoxon at f=0.75 across benchmarks (5 seeds, one Δ per benchmark).
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
from benchpress.stats import wilcoxon_per_benchmark

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _shared import (
    SEED, M_FULL, N_BENCH, OBSERVED, BENCH_IDS,
    predict_benchpress_scores, compute_prediction_error,
    holdout_half_per_benchmark, keep_only_benchmark_rows,
)
from benchpress.io_utils import write_json
from benchpress.shard_utils import (
    collect_unit_records,
    load_json_unit_shards,
    unit_shard_path,
    write_unit_json_atomic,
)

N_SEEDS = 5
DROP_RATES = [0.0, 0.25, 0.50, 0.75]
METRICS = ["medape", "medae"]
SHARDS_DIR = os.path.join(os.path.dirname(__file__), "shards")


def _precompute_units(limit_units: int | None = None) -> list[dict]:
    """Precompute all random masks in the original serial order.

    Workers consume fixed masks only; they never advance RNG state. This keeps
    the multi-shard run identical to the serial experiment design.
    """
    rng = np.random.RandomState(SEED)
    units = []
    unit_index = 0
    for j in range(N_BENCH):
        obs_rows = np.where(OBSERVED[:, j])[0]
        if len(obs_rows) < 6:
            continue
        for s in range(N_SEEDS):
            test_idx, train_idx = holdout_half_per_benchmark(j, rng, min_test=3)
            keep_by_drop = {}
            for drop_rate in DROP_RATES:
                if len(train_idx) < 2:
                    keep_by_drop[drop_rate] = []
                    continue
                n_keep = max(1, int(len(train_idx) * (1.0 - drop_rate)))
                keep_by_drop[drop_rate] = rng.permutation(len(train_idx))[:n_keep].tolist()
            units.append({
                "unit_index": unit_index,
                "bench_index": int(j),
                "bench_id": BENCH_IDS[j],
                "seed": int(s),
                "obs_rows": obs_rows.astype(int).tolist(),
                "test_idx": test_idx.astype(int).tolist(),
                "train_idx": train_idx.astype(int).tolist(),
                "keep_by_drop": {str(k): v for k, v in keep_by_drop.items()},
            })
            unit_index += 1
            if limit_units is not None and len(units) >= limit_units:
                return units
    return units


def _run_unit(unit: dict) -> dict:
    j = int(unit["bench_index"])
    seed = int(unit["seed"])
    bench_id = unit["bench_id"]
    obs_rows = np.array(unit["obs_rows"], dtype=int)
    test_idx = np.array(unit["test_idx"], dtype=int)
    train_idx = np.array(unit["train_idx"], dtype=int)
    true_scores = M_FULL[test_idx, j]

    records = []
    base_info = None
    for drop_rate in DROP_RATES:
        keep_pos = np.array(unit["keep_by_drop"][str(drop_rate)], dtype=int)
        if len(train_idx) < 2 or keep_pos.size == 0:
            if drop_rate == 0.0:
                break
            continue

        keep_idx = train_idx[keep_pos]
        M_masked = keep_only_benchmark_rows(j, keep_idx)
        P = predict_benchpress_scores(M_masked)
        pred = P[test_idx, j]
        valid = np.isfinite(pred)
        if valid.sum() < 2:
            if drop_rate == 0.0:
                break
            continue

        if drop_rate == 0.0:
            base_info = {"valid": valid, "pred": pred}
            continue

        if base_info is None:
            continue

        paired = base_info["valid"] & valid
        if paired.sum() < 2:
            continue
        mb = compute_prediction_error(true_scores[paired], base_info["pred"][paired])
        mt = compute_prediction_error(true_scores[paired], pred[paired])

        rec = {
            "bench_id": bench_id, "seed": seed, "drop_rate": drop_rate,
            "n_obs_original": int(len(obs_rows)), "n_train_kept": int(len(keep_idx)),
            "n_test": int(paired.sum()),
        }
        for mk in METRICS:
            rec[f"base_{mk}"] = float(mb[mk])
            rec[f"treat_{mk}"] = float(mt[mk])
            rec[f"delta_{mk}"] = float(mt[mk]) - float(mb[mk])
        rec["raw_base"] = [
            [int(test_idx[vi]), int(j),
             round(float(true_scores[vi]), 6),
             round(float(base_info["pred"][vi]), 6)]
            for vi in range(len(test_idx)) if paired[vi]
        ]
        rec["raw_treat"] = [
            [int(test_idx[vi]), int(j),
             round(float(true_scores[vi]), 6),
             round(float(pred[vi]), 6)]
            for vi in range(len(test_idx)) if paired[vi]
        ]
        records.append(rec)

    return {
        "unit_index": int(unit["unit_index"]),
        "bench_id": bench_id,
        "bench_index": j,
        "seed": seed,
        "records": records,
    }


def summarize_records(records: list[dict]) -> dict:
    headline = [r for r in records if r["drop_rate"] == 0.75]
    wilcoxon = wilcoxon_per_benchmark(
        headline,
        METRICS,
        invalid_median=0.0,
        invalid_p=1.0,
    )
    return {"records": records, "wilcoxon": wilcoxon}


def merge_h5_obs_count_shards(shards_dir: str) -> dict:
    units = load_json_unit_shards(shards_dir, label="H5 shard JSON files")
    records = collect_unit_records(units)
    return summarize_records(records)


def ablation_h5_obs_count(
    num_shards: int = 1,
    shard_id: int = 0,
    workers: int = 1,
    shards_dir: str = SHARDS_DIR,
    limit_units: int | None = None,
) -> dict | None:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    os.makedirs(shards_dir, exist_ok=True)
    units = _precompute_units(limit_units=limit_units)
    units = [u for u in units if int(u["unit_index"]) % num_shards == shard_id]
    pending = [
        u for u in units
        if not os.path.exists(unit_shard_path(shards_dir, "bench", u["bench_id"], int(u["seed"])))
    ]

    print(
        f"[H5] shard {shard_id}/{num_shards}: {len(units)} units, "
        f"{len(pending)} pending, workers={workers}",
        flush=True,
    )
    t0 = time.time()

    def save_unit(unit_out: dict):
        write_unit_json_atomic(
            shards_dir, "bench", unit_out["bench_id"], int(unit_out["seed"]), unit_out
        )

    done = 0
    if workers == 1:
        for unit in pending:
            save_unit(_run_unit(unit))
            done += 1
            print(
                f"  H5 unit {done:4d}/{len(pending)} "
                f"{unit['bench_id']} seed={unit['seed']} ({time.time()-t0:.0f}s)",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_unit, unit): unit for unit in pending}
            for fut in as_completed(futures):
                unit = futures[fut]
                save_unit(fut.result())
                done += 1
                print(
                    f"  H5 unit {done:4d}/{len(pending)} "
                    f"{unit['bench_id']} seed={unit['seed']} ({time.time()-t0:.0f}s)",
                    flush=True,
                )

    if num_shards == 1:
        return merge_h5_obs_count_shards(shards_dir)
    return None


def parse_args():
    default_workers = int(os.environ.get("BENCHPRESS_H4_WORKERS", "1"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--workers", type=int, default=default_workers)
    p.add_argument("--shards-dir", default=SHARDS_DIR)
    p.add_argument(
        "--output-path",
        default=os.path.join(os.path.dirname(__file__), "ablation_results.json"),
    )
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--limit-units", type=int, default=None,
                   help="Smoke-test only: precompute and run only the first N units.")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)
    if args.merge_only:
        out = merge_h5_obs_count_shards(args.shards_dir)
    else:
        out = ablation_h5_obs_count(
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            workers=args.workers,
            shards_dir=args.shards_dir,
            limit_units=args.limit_units,
        )
        if out is None:
            print(
                f"[saved shards] {args.shards_dir}; run --merge-only after all shards finish",
                flush=True,
            )
            return

    out["hypothesis"] = "H5"
    out["intervention"] = "obs_count_drop"
    write_json(args.output_path, out, indent=2)
    print(f"[saved] {args.output_path} ({len(out['records'])} records)")
    print("\n== Wilcoxon summary ==")
    for m, v in out["wilcoxon"].items():
        print(f"    {m:10s}  Δmedian={v['median_delta']:+.3f}  p={v['p_value']:.4f}  n={v['n']}")


if __name__ == "__main__":
    main()
