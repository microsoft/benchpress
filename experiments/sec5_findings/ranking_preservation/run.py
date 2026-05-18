#!/usr/bin/env python
"""Ranking preservation metrics for the BenchPress default predictor.

This script reuses the Section 4.2 prediction-first cache for the paper's
default BenchPress model (Logit Bias ALS, lambda=0.1, rank=2). It does not
rerun matrix completion. Pairwise ranking accuracy completes each benchmark
leaderboard with true seen cells plus predictions for held-out cells, then
scores same-benchmark pairs where at least one cell was held out.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

SEED = 42
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
GITHUB_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if GITHUB_ROOT not in sys.path:
    sys.path.insert(0, GITHUB_ROOT)
from benchpress.artifact_utils import ensure_default_predictions  # noqa: E402
from benchpress.io_utils import write_json_atomic  # noqa: E402
SOURCE_REL = (
    "experiments/sec4_building_benchpress/method_comparison/"
    "predictions/0124__logit__bias_als__hp01_b16f05a66b.npz"
)
SOURCE_PATH = os.path.join(GITHUB_ROOT, SOURCE_REL)
RESULTS_PATH = os.path.join(HERE, "results.json")

MARGINS = [0.0, 1.0, 2.0, 5.0]
TOP_FRACTIONS = [0.10, 0.20, 0.30]


def _load_benchmark_metadata() -> tuple[list[str], list[str], list[str], np.ndarray]:
    from benchpress.evaluation_harness import BENCH_IDS, BENCH_NAMES, BENCH_CATS, M_FULL

    bench_ids = list(BENCH_IDS)
    bench_names = [BENCH_NAMES[bid] for bid in bench_ids]
    bench_cats = [str(x) for x in BENCH_CATS]
    return bench_ids, bench_names, bench_cats, M_FULL


def _compute_ranking_accuracy(actual, predicted, heldout, margin):
    from benchpress.evaluation_harness import compute_ranking_accuracy

    return compute_ranking_accuracy(
        actual, predicted, heldout, margin=margin, aggregation="pool")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    return value


def _group_indices(fold_id: np.ndarray, test_j: np.ndarray) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (fold, bench) in enumerate(zip(fold_id.astype(int), test_j.astype(int))):
        groups[(fold, bench)].append(idx)
    return groups


def _pairwise_rows_for_group(
    fold: int,
    bench: int,
    bench_id: str,
    bench_name: str,
    bench_category: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    heldout_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    n_models = int(actual.size)

    for margin in MARGINS:
        metrics = _compute_ranking_accuracy(
            actual, predicted, heldout_mask, margin=margin)
        total = metrics["n_pairs"]
        correct = metrics["n_correct"]
        pred_ties = metrics["n_predicted_ties"]
        accuracy = metrics["accuracy"] if total else None

        rows.append({
            "fold": fold,
            "benchmark_index": bench,
            "benchmark_id": bench_id,
            "benchmark_name": bench_name,
            "benchmark_category": bench_category,
            "margin": margin,
            "n_models": n_models,
            "n_heldout_models": int(heldout_mask.sum()),
            "n_seen_models": int((~heldout_mask).sum()),
            "n_pairs": total,
            "n_correct": correct,
            "n_predicted_ties": pred_ties,
            "accuracy": accuracy,
        })
    return rows


def _top_rows_for_group(
    fold: int,
    bench: int,
    bench_id: str,
    bench_name: str,
    bench_category: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    heldout_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    n_models = int(actual.size)
    n_heldout = int(heldout_mask.sum())
    n_seen = int((~heldout_mask).sum())
    true_order = np.argsort(-actual, kind="stable")
    pred_order = np.argsort(-predicted, kind="stable")

    for frac in TOP_FRACTIONS:
        k = int(math.ceil(frac * n_models))
        k = max(1, min(k, n_models))
        true_top = set(int(x) for x in true_order[:k])
        pred_top = set(int(x) for x in pred_order[:k])
        overlap = len(true_top & pred_top)
        rows.append({
            "fold": fold,
            "benchmark_index": bench,
            "benchmark_id": bench_id,
            "benchmark_name": bench_name,
            "benchmark_category": bench_category,
            "top_fraction": frac,
            "n_models": n_models,
            "n_heldout_models": n_heldout,
            "n_seen_models": n_seen,
            "k": k,
            "overlap": overlap,
        })
    return rows


def _summarize_pairwise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for margin in MARGINS:
        subset = [r for r in rows if r["margin"] == margin and r["n_pairs"] > 0]
        total_pairs = sum(r["n_pairs"] for r in subset)
        by_benchmark: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "n_pairs": 0,
            "n_correct": 0,
        })
        for row in subset:
            bucket = by_benchmark[row["benchmark_id"]]
            bucket["n_pairs"] += row["n_pairs"]
            bucket["n_correct"] += row["n_correct"]
        benchmark_accuracies = [
            bucket["n_correct"] / bucket["n_pairs"]
            for bucket in by_benchmark.values()
            if bucket["n_pairs"] > 0
        ]
        summary[str(margin)] = {
            "n_groups": len(subset),
            "n_benchmarks": len(by_benchmark),
            "n_pairs": total_pairs,
            "accuracy": (
                float(np.median(benchmark_accuracies))
                if benchmark_accuracies else None
            ),
        }
    return summary


def _summarize_top(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for frac in TOP_FRACTIONS:
        subset = [r for r in rows if r["top_fraction"] == frac]
        total_k = sum(r["k"] for r in subset)
        by_benchmark: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "total_k": 0,
            "overlap": 0,
        })
        for row in subset:
            bucket = by_benchmark[row["benchmark_id"]]
            bucket["total_k"] += row["k"]
            bucket["overlap"] += row["overlap"]
        benchmark_recoveries = [
            bucket["overlap"] / bucket["total_k"]
            for bucket in by_benchmark.values()
            if bucket["total_k"] > 0
        ]
        summary[str(frac)] = {
            "n_groups": len(subset),
            "n_benchmarks": len(by_benchmark),
            "total_k": total_k,
            "recovery": (
                float(np.median(benchmark_recoveries))
                if benchmark_recoveries else None
            ),
        }
    return summary


def main() -> None:
    if not os.path.exists(SOURCE_PATH):
        ensure_default_predictions()

    bench_ids, bench_names, bench_cats, M_full = _load_benchmark_metadata()

    with np.load(SOURCE_PATH, allow_pickle=False) as data:
        fold_id = data["fold_id"].astype(int)
        test_i = data["test_i"].astype(int)
        test_j = data["test_j"].astype(int)
        predicted_all = data["predicted"].astype(float)
        source_metadata = json.loads(str(data["metadata_json"]))

    pairwise_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    for (fold, bench), indices in sorted(_group_indices(fold_id, test_j).items()):
        idx = np.asarray(indices, dtype=int)

        observed_models = np.where(np.isfinite(M_full[:, bench]))[0]
        if int(observed_models.size) < 2:
            continue

        completed = M_full[observed_models, bench].astype(float, copy=True)
        heldout_mask = np.zeros(int(observed_models.size), dtype=bool)
        obs_pos = {int(model): pos for pos, model in enumerate(observed_models)}
        for row_idx in idx:
            model = int(test_i[row_idx])
            pos = obs_pos.get(model)
            if pos is None:
                continue
            completed[pos] = predicted_all[row_idx]
            heldout_mask[pos] = True

        valid_completed = (
            np.isfinite(M_full[observed_models, bench])
            & np.isfinite(completed)
        )
        if int(valid_completed.sum()) < 2 or not bool(heldout_mask[valid_completed].any()):
            continue
        pairwise_actual = M_full[observed_models, bench][valid_completed]
        pairwise_predicted = completed[valid_completed]
        pairwise_heldout = heldout_mask[valid_completed]

        pairwise_rows.extend(_pairwise_rows_for_group(
            fold, bench, bench_ids[bench], bench_names[bench], bench_cats[bench],
            pairwise_actual, pairwise_predicted, pairwise_heldout))

        top_rows.extend(_top_rows_for_group(
            fold, bench, bench_ids[bench], bench_names[bench], bench_cats[bench],
            pairwise_actual, pairwise_predicted, pairwise_heldout))

    output = {
        "metadata": {
            "experiment": "sec5_findings/ranking_preservation",
            "source_prediction_cache": SOURCE_REL,
            "source_metadata": source_metadata,
            "margins": MARGINS,
            "top_fractions": TOP_FRACTIONS,
            "metric_definitions": {
                "pairwise_accuracy": (
                    "for each fold and benchmark, complete the benchmark leaderboard "
                    "by using true observed scores for seen cells and BenchPress "
                    "predictions for held-out cells; among all same-benchmark model "
                    "pairs, discard pairs where both cells were seen and score the "
                    "remaining pairs whose true score gap clears the margin; summary "
                    "accuracy is the median across benchmarks"
                ),
                "margin": "minimum absolute true score gap required for a pair",
                "top_fraction_recovery": (
                    "for each fold and benchmark, complete the full observed "
                    "leaderboard by using true scores for seen cells and BenchPress "
                    "predictions for held-out cells; compare the true and completed "
                    "top-k model sets on that full observed leaderboard; summary "
                    "recovery is the median across benchmarks"
                ),
            },
        },
        "pairwise_rows": pairwise_rows,
        "top_rows": top_rows,
        "summary": {
            "pairwise_by_margin": _summarize_pairwise(pairwise_rows),
            "top_by_fraction": _summarize_top(top_rows),
        },
    }

    write_json_atomic(
        RESULTS_PATH, _json_safe(output),
        indent=2, sort_keys=True, trailing_newline=True,
    )
    print(f"Wrote {RESULTS_PATH}")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
