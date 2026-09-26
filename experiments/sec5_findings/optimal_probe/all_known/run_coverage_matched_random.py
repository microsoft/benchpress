#!/usr/bin/env python3
"""Coverage-matched random probe subsets for the all-known hidden-only protocol."""

import argparse
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    load_benchmark_allowlist,
)
from benchpress.io_utils import load_json, write_json_atomic
from evaluate_order import DEFAULT_ORDER_JSON, RESULTS_DIR, _evaluate_k, _manifest

DEFAULT_ALLOWLIST = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "candidate_allowlists", "user_cheap_20260505.json",
))
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "coverage_matched_random_hidden_only.json.gz")
PROTOCOL = "coverage_matched_random_all_known_hidden_only_v1"
DEFAULT_K_VALUES = (1, 3, 5, 10)


def _observed_cell_count(benchmark_indices):
    """Return the number of observed matrix cells in the selected benchmarks."""
    indices = [int(j) for j in benchmark_indices]
    return int(OBSERVED[:, indices].sum()) if indices else 0


def _observed_counts(benchmark_indices):
    """Return observed-cell counts for candidate benchmark indices."""
    return np.array([int(OBSERVED[:, int(j)].sum()) for j in benchmark_indices], dtype=float)


def _load_greedy_targets(order_json_path):
    """Return coverage targets for the MedAE greedy prefixes used for matching."""
    payload = load_json(order_json_path)
    orderings = payload.get("orderings", {})
    targets = {}
    for pool_key, order_key in (("any", "medae_any"), ("low_cost", "medae_low_cost")):
        if order_key not in orderings:
            raise SystemExit(f"{order_json_path} is missing ordering {order_key!r}")
        indices = [BENCH_IDS.index(bid) for bid in orderings[order_key]["benchmark_ids"]]
        targets[pool_key] = {
            "order_key": order_key,
            "order_label": orderings[order_key].get("label"),
            "order_benchmark_ids": orderings[order_key]["benchmark_ids"],
            "revealed_cells_by_k": {
                int(k): _observed_cell_count(indices[:int(k)])
                for k in range(1, len(indices) + 1)
            },
        }
    return targets


def _pool_indices(pool_key, low_cost_allowlist):
    """Return candidate benchmark indices for one random-subset pool."""
    if pool_key == "any":
        return list(range(N_BENCH)), BENCH_IDS
    if pool_key == "low_cost":
        allowlist_indices, allowlist_ids = load_benchmark_allowlist(
            low_cost_allowlist, label="Low-cost candidate allowlist",
        )
        ordered_ids = [bid for bid in allowlist_ids if BENCH_IDS.index(bid) in allowlist_indices]
        return [BENCH_IDS.index(bid) for bid in ordered_ids], ordered_ids
    raise ValueError(f"Unknown pool key: {pool_key!r}")


def _draw_matched_subsets(
    pool_key,
    candidate_indices,
    k,
    target_revealed_cells,
    n_subsets,
    tolerance_fraction,
    seed,
    max_draws,
    proposal_power,
):
    """Sample unique k-subsets whose observed-cell count matches the target band."""
    if k > len(candidate_indices):
        raise ValueError(f"k={k} exceeds {pool_key} candidate pool size {len(candidate_indices)}")
    rng = np.random.RandomState(int(seed))
    candidate_indices = [int(j) for j in candidate_indices]
    weights = np.power(_observed_counts(candidate_indices) + 1.0, float(proposal_power))
    weights = weights / weights.sum()
    lower = int(np.floor((1.0 - tolerance_fraction) * target_revealed_cells))
    upper = int(np.ceil((1.0 + tolerance_fraction) * target_revealed_cells))
    seen = set()
    accepted = []
    draws = 0
    while len(accepted) < n_subsets and draws < max_draws:
        draws += 1
        sample = rng.choice(candidate_indices, size=int(k), replace=False, p=weights)
        key = tuple(sorted(int(j) for j in sample))
        if key in seen:
            continue
        seen.add(key)
        revealed_cells = _observed_cell_count(key)
        if lower <= revealed_cells <= upper:
            accepted.append({
                "pool": pool_key,
                "k": int(k),
                "subset_id": f"{pool_key}_k{k}_s{len(accepted):03d}",
                "unique_subset_id": f"{pool_key}_k{k}_u{len(accepted):03d}",
                "is_reused_subset": False,
                "replicate_index": 0,
                "seed": int(seed),
                "draw_index": int(draws),
                "probe_indices": [int(j) for j in sample],
                "probe_ids": [BENCH_IDS[int(j)] for j in sample],
                "probe_names": [
                    BENCH_NAMES.get(BENCH_IDS[int(j)], BENCH_IDS[int(j)])
                    for j in sample
                ],
                "revealed_cells": int(revealed_cells),
                "target_revealed_cells": int(target_revealed_cells),
                "tolerance_low": int(lower),
                "tolerance_high": int(upper),
            })
    if len(accepted) < n_subsets:
        if not accepted:
            raise RuntimeError(
                f"Accepted 0 {pool_key} k={k} subsets after {draws} draws; "
                f"target={target_revealed_cells}, band=[{lower}, {upper}]"
            )
        unique = list(accepted)
        next_idx = len(accepted)
        while len(accepted) < n_subsets:
            source = unique[(len(accepted) - len(unique)) % len(unique)]
            clone = dict(source)
            clone["subset_id"] = f"{pool_key}_k{k}_s{next_idx:03d}"
            clone["is_reused_subset"] = True
            clone["replicate_index"] = 1 + (len(accepted) - len(unique)) // len(unique)
            accepted.append(clone)
            next_idx += 1
    return accepted, draws, len(seen)


def _finite_iqr(rows, field):
    """Return median and interquartile range for finite numeric row fields."""
    vals = np.array([
        float(row[field])
        for row in rows
        if row.get(field) is not None and np.isfinite(float(row[field]))
    ], dtype=float)
    if len(vals) == 0:
        return {"median": None, "q25": None, "q75": None}
    return {
        "median": float(np.median(vals)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
    }


def _evaluate_subset(unit):
    """Evaluate one sampled subset and attach subset metadata to rows and summaries."""
    subset, rank = unit
    evaluated = _evaluate_k((int(subset["k"]), subset["probe_indices"], int(rank)))
    raw_predictions = evaluated.pop("raw_predictions")
    hidden = evaluated["hidden_only"]
    with_probe = evaluated["with_probe_zero"]
    summary = {
        "pool": subset["pool"],
        "k": int(subset["k"]),
        "subset_id": subset["subset_id"],
        "unique_subset_id": subset["unique_subset_id"],
        "is_reused_subset": bool(subset["is_reused_subset"]),
        "replicate_index": int(subset["replicate_index"]),
        "seed": int(subset["seed"]),
        "draw_index": int(subset["draw_index"]),
        "probe_ids": subset["probe_ids"],
        "probe_names": subset["probe_names"],
        "revealed_cells": int(subset["revealed_cells"]),
        "target_revealed_cells": int(subset["target_revealed_cells"]),
        "tolerance_low": int(subset["tolerance_low"]),
        "tolerance_high": int(subset["tolerance_high"]),
        "hidden_cells": int(hidden["total_cells"]),
        "hidden_finite_n": int(hidden["n"]),
        "hidden_only": hidden,
        "with_probe_zero": with_probe,
    }
    for row in raw_predictions:
        row["pool"] = subset["pool"]
        row["subset_id"] = subset["subset_id"]
        row["unique_subset_id"] = subset["unique_subset_id"]
        row["is_reused_subset"] = bool(subset["is_reused_subset"])
        row["seed"] = int(subset["seed"])
    return summary, raw_predictions


def _clone_evaluation(summary, raw_predictions, subset):
    """Copy one unique-subset evaluation onto one reported subset replicate."""
    cloned_summary = copy.deepcopy(summary)
    cloned_summary["subset_id"] = subset["subset_id"]
    cloned_summary["is_reused_subset"] = bool(subset["is_reused_subset"])
    cloned_summary["replicate_index"] = int(subset["replicate_index"])
    cloned_raw = []
    for row in raw_predictions:
        new_row = dict(row)
        new_row["subset_id"] = subset["subset_id"]
        new_row["is_reused_subset"] = bool(subset["is_reused_subset"])
        cloned_raw.append(new_row)
    return cloned_summary, cloned_raw


def _aggregate_by_pool_k(summaries):
    """Aggregate subset summaries by pool and k using medians and IQRs."""
    grouped = {}
    for row in summaries:
        grouped.setdefault((row["pool"], int(row["k"])), []).append(row)
    output = []
    for pool_key, k in sorted(grouped, key=lambda item: (item[0][0], item[0][1])):
        rows = grouped[(pool_key, k)]
        flat = []
        for row in rows:
            hidden = row["hidden_only"]
            flat.append({
                "revealed_cells": row["revealed_cells"],
                "hidden_cells": row["hidden_cells"],
                "medae": hidden["medae"],
                "medape": hidden["medape"],
                "p90_abs_error": hidden["p90_abs_error"],
            })
        output.append({
            "pool": pool_key,
            "k": int(k),
            "n_subsets": len(rows),
            "target_revealed_cells": int(rows[0]["target_revealed_cells"]),
            "revealed_cells": _finite_iqr(flat, "revealed_cells"),
            "hidden_cells": _finite_iqr(flat, "hidden_cells"),
            "medae": _finite_iqr(flat, "medae"),
            "medape": _finite_iqr(flat, "medape"),
            "p90_abs_error": _finite_iqr(flat, "p90_abs_error"),
        })
    return output


def _resolve_output_path(output_arg):
    """Resolve an output path under results/ unless an absolute path is supplied."""
    if os.path.isabs(output_arg):
        return output_arg
    return os.path.join(RESULTS_DIR, output_arg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-order-json", default=DEFAULT_ORDER_JSON)
    parser.add_argument("--low-cost-allowlist", default=DEFAULT_ALLOWLIST)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--n-subsets", type=int, default=10)
    parser.add_argument("--tolerance-fraction", type=float, default=0.10)
    parser.add_argument("--base-seed", type=int, default=20260925)
    parser.add_argument("--max-draws-per-k", type=int, default=200000)
    parser.add_argument("--coverage-proposal-power", type=float, default=2.0,
                        help="Draw proposals with probability proportional to observed_count**power.")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.n_subsets < 1:
        raise SystemExit("--n-subsets must be positive")
    if not 0.0 <= args.tolerance_fraction < 1.0:
        raise SystemExit("--tolerance-fraction must be in [0, 1)")

    targets = _load_greedy_targets(args.fixed_order_json)
    pool_configs = {}
    for pool_key in ("any", "low_cost"):
        indices, ids = _pool_indices(pool_key, args.low_cost_allowlist)
        pool_configs[pool_key] = {
            "candidate_benchmark_ids": ids,
            "candidate_pool_size": len(indices),
            "candidate_indices": indices,
            "candidate_observed_counts": {
                BENCH_IDS[int(j)]: int(OBSERVED[:, int(j)].sum())
                for j in indices
            },
        }

    subsets = []
    draw_diagnostics = []
    for pool_offset, pool_key in enumerate(("any", "low_cost")):
        candidate_indices = pool_configs[pool_key]["candidate_indices"]
        for k in args.k_values:
            k = int(k)
            target_by_k = targets[pool_key]["revealed_cells_by_k"]
            if k not in target_by_k:
                raise SystemExit(
                    f"k={k} is unavailable for greedy order {targets[pool_key]['order_key']}"
                )
            accepted, draws, unique_draws = _draw_matched_subsets(
                pool_key=pool_key,
                candidate_indices=candidate_indices,
                k=k,
                target_revealed_cells=target_by_k[k],
                n_subsets=args.n_subsets,
                tolerance_fraction=args.tolerance_fraction,
                seed=args.base_seed + 1009 * pool_offset + 37 * k,
                max_draws=args.max_draws_per_k,
                proposal_power=args.coverage_proposal_power,
            )
            subsets.extend(accepted)
            draw_diagnostics.append({
                "pool": pool_key,
                "k": k,
                "draws": int(draws),
                "unique_draws": int(unique_draws),
                "accepted": len(accepted),
                "unique_accepted": len({row["unique_subset_id"] for row in accepted}),
                "reused_to_reach_n_subsets": any(row["is_reused_subset"] for row in accepted),
            })

    unique_subsets = {}
    for subset in subsets:
        unique_subsets.setdefault(subset["unique_subset_id"], subset)
    print(
        f"Prepared {len(subsets)} reported subsets from {len(unique_subsets)} unique subsets",
        flush=True,
    )

    unique_units = [(subset, args.rank) for subset in unique_subsets.values()]
    evaluated_by_unique = {}
    if args.workers <= 1:
        for idx, unit in enumerate(unique_units, start=1):
            summary, raw = _evaluate_subset(unit)
            evaluated_by_unique[summary["unique_subset_id"]] = (summary, raw)
            print(f"Evaluated {idx}/{len(unique_units)} unique subsets", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(unique_units))) as pool:
            futures = [pool.submit(_evaluate_subset, unit) for unit in unique_units]
            for idx, future in enumerate(as_completed(futures), start=1):
                summary, raw = future.result()
                evaluated_by_unique[summary["unique_subset_id"]] = (summary, raw)
                print(f"Evaluated {idx}/{len(unique_units)} unique subsets", flush=True)

    subset_summaries = []
    raw_predictions = []
    for subset in subsets:
        summary, raw = evaluated_by_unique[subset["unique_subset_id"]]
        cloned_summary, cloned_raw = _clone_evaluation(summary, raw, subset)
        subset_summaries.append(cloned_summary)
        raw_predictions.extend(cloned_raw)
    subset_summaries.sort(key=lambda row: (row["pool"], row["k"], row["subset_id"]))
    raw_predictions.sort(key=lambda row: (
        row["pool"], row["k"], row["subset_id"],
        row["target_model_index"], row["benchmark_index"],
    ))

    output = {
        "config": {
            "protocol": PROTOCOL,
            "k_values": [int(k) for k in args.k_values],
            "n_subsets_per_pool_k": int(args.n_subsets),
            "tolerance_fraction": float(args.tolerance_fraction),
            "base_seed": int(args.base_seed),
            "rank": int(args.rank),
            "fixed_order_json": os.path.relpath(
                os.path.abspath(os.path.expanduser(args.fixed_order_json)),
                REPO_ROOT,
            ),
            "low_cost_allowlist": os.path.relpath(
                os.path.abspath(os.path.expanduser(args.low_cost_allowlist)),
                REPO_ROOT,
            ),
            "target_orders": targets,
            "candidate_pools": {
                key: {
                    "candidate_pool_size": value["candidate_pool_size"],
                    "candidate_benchmark_ids": value["candidate_benchmark_ids"],
                    "candidate_observed_counts": value["candidate_observed_counts"],
                }
                for key, value in pool_configs.items()
            },
            "n_models": int(N_MODELS),
            "n_bench": int(N_BENCH),
            "n_observed": int(OBSERVED.sum()),
            "prediction_engine": (
                f"predict_logit_bias_als_scores (Logit Bias ALS, rank={args.rank}, "
                "lambda=0.1)"
            ),
            "matching_rule": (
                "For each pool and k, draw random k-subset proposals without replacement "
                "from the candidate pool, using probability proportional to each benchmark's "
                "observed-cell count raised to coverage_proposal_power, and keep unique "
                "subsets whose observed probe cells are within +/- tolerance_fraction of "
                "the corresponding MedAE greedy prefix's observed probe cells. The proposal "
                "only makes the high-coverage conditional sample tractable; every retained "
                "subset must pass the coverage-matching band. If fewer than n_subsets unique "
                "matches exist after max_draws_per_k, reuse the matched subsets with "
                "is_reused_subset=true so every pool-k has the same reported replicate "
                "count without hiding the unique-subset count."
            ),
            "coverage_proposal_power": float(args.coverage_proposal_power),
            "eval_scope": "hidden_only excludes revealed probe cells; with_probe_zero keeps them as exact",
        },
        "manifest": _manifest(),
        "draw_diagnostics": draw_diagnostics,
        "subset_summaries": subset_summaries,
        "summary_by_pool_k": _aggregate_by_pool_k(subset_summaries),
        "raw_predictions": raw_predictions,
    }

    output_path = _resolve_output_path(args.out)
    write_json_atomic(output_path, output, indent=2)
    print(f"Saved -> {output_path}")
    for row in output["summary_by_pool_k"]:
        medae = row["medae"]["median"]
        p90 = row["p90_abs_error"]["median"]
        revealed = row["revealed_cells"]["median"]
        print(
            f"{row['pool']} k={row['k']}: revealed median={revealed:.0f}, "
            f"MedAE median={medae:.3f}, P90 median={p90:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
