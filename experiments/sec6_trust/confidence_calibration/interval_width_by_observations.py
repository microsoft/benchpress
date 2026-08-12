#!/usr/bin/env python
"""Controlled CI-width analysis by target-benchmark observation count.

Reviewer rdwq asks how prediction intervals change as more conditioning scores
are observed. This script fixes the benchmark universe and target cells before
varying the number of revealed observations, so interval widths across k are
comparable.

Protocol:
  * Select benchmark columns with at least 30 observed model scores.
  * For each eligible benchmark and seed, hold out the same 10 observed scores.
  * Reveal k in {5, 10, 15, 20} nested conditioning scores from the remaining
    observations in that same benchmark column.
  * Hide every other observed score in the target benchmark column.
  * Keep all other benchmark columns observed.
  * Predict the fixed 10 test scores with the fixed BenchPress point predictor.
  * Build leave-one-benchmark-out conformal intervals separately for each k.

Outputs are written next to this file unless --outdir is provided.
"""

import argparse
import math
import os

import numpy as np

from benchpress.evaluation_harness import (
    M_FULL,
    BENCH_IDS,
    BENCH_NAMES,
)
from benchpress.io_utils import write_json_atomic, write_npz_compressed_atomic
from benchpress.methods.predictors import predict_benchpress_scores

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COUNTS = (5, 10, 15, 20)
DEFAULT_CI = 0.80
DEFAULT_N_SEEDS = 10
DEFAULT_BASE_SEED = 42
DEFAULT_TEST_COUNT = 10
DEFAULT_MIN_OBSERVATIONS = 30
JSON_NAME = "interval_width_by_observations.json"
MD_NAME = "interval_width_by_observations.md"
RAW_NAME = "interval_width_by_observations_raw.npz"


def _parse_counts(value):
    counts = tuple(int(x) for x in value.split(",") if x.strip())
    if not counts:
        raise argparse.ArgumentTypeError("conditioning counts cannot be empty")
    if any(k <= 0 for k in counts):
        raise argparse.ArgumentTypeError("conditioning counts must be positive")
    if tuple(sorted(counts)) != counts:
        raise argparse.ArgumentTypeError("conditioning counts must be sorted")
    return counts


def eligible_benchmarks(min_observations, test_count, conditioning_counts):
    """Benchmark columns with enough observed cells for fixed test plus max-k context."""
    required = max(int(min_observations), int(test_count) + max(conditioning_counts))
    rows = []
    for j in range(M_FULL.shape[1]):
        observed_rows = np.flatnonzero(np.isfinite(M_FULL[:, j]))
        if observed_rows.size >= required:
            rows.append({
                "bench_index": int(j),
                "bench_id": BENCH_IDS[j],
                "bench_name": BENCH_NAMES[BENCH_IDS[j]],
                "n_observed": int(observed_rows.size),
            })
    return rows


def _conformal_half_width(errors, ci):
    """Finite-sample split-conformal absolute-error quantile."""
    errors = np.sort(np.asarray(errors, dtype=float))
    if errors.size == 0:
        return float("nan")
    rank = int(math.ceil((errors.size + 1) * float(ci))) - 1
    rank = min(max(rank, 0), errors.size - 1)
    return float(errors[rank])


def _run_predictions(eligible, conditioning_counts, n_seeds, base_seed, test_count):
    seed_idx, bench_idx, model_idx, conditioning_count = [], [], [], []
    actual, predicted = [], []

    for seed_offset in range(int(n_seeds)):
        rng = np.random.RandomState(int(base_seed) + seed_offset)
        for bench in eligible:
            j = int(bench["bench_index"])
            observed_rows = np.flatnonzero(np.isfinite(M_FULL[:, j]))
            shuffled = rng.permutation(observed_rows)
            test_rows = shuffled[:test_count]
            conditioning_order = shuffled[test_count:]

            for k in conditioning_counts:
                condition_rows = conditioning_order[:k]
                M_train = np.array(M_FULL, copy=True)
                M_train[observed_rows, j] = np.nan
                M_train[condition_rows, j] = M_FULL[condition_rows, j]

                M_pred = predict_benchpress_scores(M_train)
                for i in test_rows:
                    seed_idx.append(seed_offset)
                    bench_idx.append(j)
                    model_idx.append(int(i))
                    conditioning_count.append(int(k))
                    actual.append(float(M_FULL[i, j]))
                    predicted.append(float(M_pred[i, j]))

    return {
        "seed_idx": np.asarray(seed_idx, dtype=np.int16),
        "bench_idx": np.asarray(bench_idx, dtype=np.int16),
        "model_idx": np.asarray(model_idx, dtype=np.int16),
        "conditioning_count": np.asarray(conditioning_count, dtype=np.int16),
        "actual": np.asarray(actual, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    }


def _add_leave_benchmark_out_intervals(arrays, ci):
    errors = np.abs(arrays["predicted"] - arrays["actual"])
    half_width = np.full(errors.shape, np.nan, dtype=float)

    for k in np.unique(arrays["conditioning_count"]):
        k_mask = arrays["conditioning_count"] == k
        for j in np.unique(arrays["bench_idx"][k_mask]):
            target = k_mask & (arrays["bench_idx"] == j)
            calibration = k_mask & (arrays["bench_idx"] != j)
            half_width[target] = _conformal_half_width(errors[calibration], ci)

    lower = arrays["predicted"] - half_width
    upper = arrays["predicted"] + half_width
    covered = (arrays["actual"] >= lower) & (arrays["actual"] <= upper)
    out = dict(arrays)
    out.update({
        "abs_error": errors,
        "half_width": half_width,
        "interval_width": 2.0 * half_width,
        "lower": lower,
        "upper": upper,
        "covered": covered.astype(np.int8),
    })
    return out


def _summary_rows(arrays, conditioning_counts):
    rows = []
    for k in conditioning_counts:
        mask = arrays["conditioning_count"] == k
        if not np.any(mask):
            continue
        rows.append({
            "conditioning_scores": int(k),
            "n_predictions": int(mask.sum()),
            "n_benchmarks": int(np.unique(arrays["bench_idx"][mask]).size),
            "n_seeds": int(np.unique(arrays["seed_idx"][mask]).size),
            "medae": float(np.median(arrays["abs_error"][mask])),
            "coverage": float(np.mean(arrays["covered"][mask])),
            "median_interval_width": float(np.nanmedian(arrays["interval_width"][mask])),
            "p25_interval_width": float(np.nanpercentile(arrays["interval_width"][mask], 25)),
            "p75_interval_width": float(np.nanpercentile(arrays["interval_width"][mask], 75)),
        })
    return rows


def _markdown_table(rows, ci):
    ci_label = f"{int(round(float(ci) * 100))}%"
    lines = [
        f"| Revealed scores per benchmark | Held-out predictions | Benchmarks | MedAE | {ci_label} coverage | Median interval width |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['conditioning_scores']} | {row['n_predictions']:,} | "
            f"{row['n_benchmarks']} | {row['medae']:.2f} | "
            f"{row['coverage']:.3f} | {row['median_interval_width']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=SCRIPT_DIR)
    parser.add_argument("--conditioning-counts", type=_parse_counts, default=DEFAULT_COUNTS)
    parser.add_argument("--ci", type=float, default=DEFAULT_CI)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--min-observations", type=int, default=DEFAULT_MIN_OBSERVATIONS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    conditioning_counts = tuple(args.conditioning_counts)
    eligible = eligible_benchmarks(
        args.min_observations,
        args.test_count,
        conditioning_counts,
    )
    if args.smoke:
        eligible = eligible[:3]
        n_seeds = 1
    else:
        n_seeds = args.n_seeds
    if not eligible:
        raise ValueError("No eligible benchmarks under the requested protocol.")

    arrays = _run_predictions(
        eligible,
        conditioning_counts,
        n_seeds=n_seeds,
        base_seed=args.base_seed,
        test_count=args.test_count,
    )
    arrays = _add_leave_benchmark_out_intervals(arrays, args.ci)
    rows = _summary_rows(arrays, conditioning_counts)

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, RAW_NAME)
    json_path = os.path.join(args.outdir, JSON_NAME)
    md_path = os.path.join(args.outdir, MD_NAME)
    write_npz_compressed_atomic(raw_path, **arrays)
    payload = {
        "setting": {
            "matrix_shape": [int(M_FULL.shape[0]), int(M_FULL.shape[1])],
            "eligible_rule": (
                "benchmark columns with at least max(min_observations, "
                "test_count + max(conditioning_counts)) observed scores"
            ),
            "min_observations": int(args.min_observations),
            "test_count": int(args.test_count),
            "conditioning_counts": [int(k) for k in conditioning_counts],
            "ci": float(args.ci),
            "n_seeds": int(n_seeds),
            "base_seed": int(args.base_seed),
            "smoke": bool(args.smoke),
            "prediction_method": "BenchPress = Logit Bias ALS, rank=2, lambda=0.1",
            "interval_method": (
                "leave-one-benchmark-out conformal absolute-error intervals, "
                "computed separately for each conditioning count"
            ),
            "fixed_target_rule": (
                "for each benchmark and seed, the same 10 test cells are used "
                "for all conditioning counts; conditioning cells are nested prefixes"
            ),
        },
        "n_eligible_benchmarks": int(len(eligible)),
        "eligible_benchmarks": eligible,
        "summary": rows,
    }
    write_json_atomic(json_path, payload, indent=2, sort_keys=True, trailing_newline=True)

    md = _markdown_table(rows, args.ci)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")
    print(f"wrote: {raw_path}")


if __name__ == "__main__":
    main()
