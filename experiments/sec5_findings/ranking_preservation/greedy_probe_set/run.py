#!/usr/bin/env python3
"""Greedy probe-set selection for margin-5 pairwise ranking accuracy.

This is a Section 5.2 ranking-preservation experiment. It reuses the Section
5.1 all-known probe evaluator as a prediction primitive, but the greedy
objective is ranking preservation: choose the next probe benchmark that
maximizes margin-5 pairwise ranking accuracy over the fixed all-known-cell
universe.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import (  # noqa: E402
    BENCH_IDS,
    BENCH_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    compute_ranking_accuracy,
    evaluate_probe_set,
    load_benchmark_allowlist,
    pack_probe_predictions,
    probe_candidate_cache_path,
)
from benchpress.all_methods import predict_benchpress_scores  # noqa: E402
from benchpress.io_utils import load_json, safe_token, write_json_atomic  # noqa: E402
from benchpress.shard_utils import short_text_hash  # noqa: E402

SEED = 42
MARGIN = 5.0
MAX_STEPS = 10
EVAL_PROTOCOL = "all_known_probe_cells_zero_error_v1"
OBJECTIVE = "margin5_pairwise_ranking_accuracy"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _ranking_metrics(
    predictions: list[tuple[int, int, float, float]],
    margin: float = MARGIN,
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot compute ranking accuracy with zero predictions")

    arr = np.asarray(predictions, dtype=object)
    actual = arr[:, 2].astype(float)
    predicted = arr[:, 3].astype(float)
    bench_idx = arr[:, 1].astype(int)

    # In the all-known probe-set setting, every observed cell is in the evaluation
    # universe. Probe cells are exact predictions, so their pairs are counted
    # rather than removed from the denominator.
    evaluated = np.ones(len(actual), dtype=bool)
    grouped = compute_ranking_accuracy(
        actual,
        predicted,
        evaluated,
        groups=bench_idx,
        margin=margin,
        aggregation="per_group_median",
    )
    pooled = compute_ranking_accuracy(
        actual,
        predicted,
        evaluated,
        margin=margin,
        aggregation="pool",
    )

    per_benchmark = {}
    for j, metrics in grouped["per_group"].items():
        j_int = int(j)
        per_benchmark[BENCH_IDS[j_int]] = {
            "accuracy": (
                float(metrics["accuracy"])
                if np.isfinite(metrics["accuracy"]) else None
            ),
            "n_pairs": int(metrics["n_pairs"]),
            "n_correct": int(metrics["n_correct"]),
            "n_predicted_ties": int(metrics["n_predicted_ties"]),
        }

    accuracy = grouped["accuracy_median"]
    return {
        "margin": float(margin),
        "pairwise_accuracy_margin5": (
            float(accuracy) if np.isfinite(accuracy) else None
        ),
        "score": (
            float(1.0 - accuracy) if np.isfinite(accuracy) else float("inf")
        ),
        "n_pairs": int(grouped["n_pairs"]),
        "n_correct": int(grouped["n_correct"]),
        "n_predicted_ties": int(grouped["n_predicted_ties"]),
        "pooled_accuracy": (
            float(pooled["accuracy"]) if np.isfinite(pooled["accuracy"]) else None
        ),
        "pooled_n_pairs": int(pooled["n_pairs"]),
        "pooled_n_correct": int(pooled["n_correct"]),
        "per_benchmark": per_benchmark,
    }


def _evaluate_candidate(args):
    probe_set, cand_j = args
    predictions, score_metrics, _ = evaluate_probe_set(
        probe_set,
        predict_benchpress_scores,
        metric="medape",
    )
    ranking = _ranking_metrics(predictions, margin=MARGIN)
    return cand_j, predictions, score_metrics, ranking


def _cache_root(out_path: str, candidate_allowlist_ids: list[str] | None) -> str:
    stem = os.path.basename(out_path)
    if stem.endswith(".json.gz"):
        stem = stem[:-len(".json.gz")]
    else:
        stem = os.path.splitext(stem)[0]
    name = (
        f"{safe_token(stem)}__objective-{safe_token(OBJECTIVE)}"
        f"__protocol-{safe_token(EVAL_PROTOCOL)}"
    )
    if candidate_allowlist_ids is not None:
        digest = short_text_hash("\n".join(candidate_allowlist_ids), n=12)
        name += f"__allowlist-{digest}"
    return os.path.join(RESULTS_DIR, ".candidate_cache", name)


def _load_candidate_cache(
    path: str,
    expected_benchmark_id: str,
    expected_probe_set: list[str],
) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    payload = load_json(path)
    if payload.get("objective") != OBJECTIVE:
        raise RuntimeError(f"Candidate cache objective mismatch in {path}")
    if payload.get("eval_protocol") != EVAL_PROTOCOL:
        raise RuntimeError(f"Candidate cache protocol mismatch in {path}")
    if payload.get("probe_set_before_candidate") != expected_probe_set:
        raise RuntimeError(f"Candidate cache probe-set mismatch in {path}")
    record = payload.get("record", {})
    if record.get("benchmark_id") != expected_benchmark_id:
        raise RuntimeError(f"Candidate cache benchmark mismatch in {path}")
    return record


def _candidate_record(
    cand_j: int,
    predictions: list[tuple[int, int, float, float]],
    score_metrics: dict[str, Any],
    ranking: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark_id": BENCH_IDS[cand_j],
        "benchmark_name": BENCH_NAMES.get(BENCH_IDS[cand_j], BENCH_IDS[cand_j]),
        "score": ranking["score"],
        "pairwise_accuracy_margin5": ranking["pairwise_accuracy_margin5"],
        "n_pairs": ranking["n_pairs"],
        "n_correct": ranking["n_correct"],
        "n_predicted_ties": ranking["n_predicted_ties"],
        "pooled_accuracy": ranking["pooled_accuracy"],
        "pooled_n_pairs": ranking["pooled_n_pairs"],
        "pooled_n_correct": ranking["pooled_n_correct"],
        "medape": (
            float(score_metrics["medape"])
            if np.isfinite(score_metrics["medape"]) else None
        ),
        "medae": (
            float(score_metrics["medae"])
            if np.isfinite(score_metrics["medae"]) else None
        ),
        "per_benchmark_ranking": ranking["per_benchmark"],
        "predictions": pack_probe_predictions(predictions),
    }


def _init_worker(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--out",
        type=str,
        default="greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz",
    )
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--candidate-allowlist", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(SEED)
    random.seed(SEED)

    candidate_allowlist, candidate_allowlist_ids = load_benchmark_allowlist(
        args.candidate_allowlist, label="Candidate allowlist",
    )
    candidates = list(range(N_BENCH))
    if candidate_allowlist is not None:
        candidates = [j for j in candidates if j in candidate_allowlist]
    if args.candidate_limit is not None:
        candidates = candidates[:args.candidate_limit]
    if not candidates:
        raise RuntimeError("Candidate universe is empty")

    out_path = os.path.join(RESULTS_DIR, args.out)
    selected: list[int] = []
    remaining = list(candidates)
    trajectory: list[dict[str, Any]] = []

    if os.path.exists(out_path):
        prev = load_json(out_path)
        expected_config = {
            "objective": OBJECTIVE,
            "margin": MARGIN,
            "eval_protocol": EVAL_PROTOCOL,
            "candidate_allowlist_path": (
                os.path.relpath(args.candidate_allowlist, REPO_ROOT)
                if args.candidate_allowlist else None
            ),
            "candidate_allowlist_ids": candidate_allowlist_ids,
            "candidate_limit": args.candidate_limit,
            "n_candidates": len(candidates),
        }
        prev_config = {k: prev.get("config", {}).get(k) for k in expected_config}
        if prev_config != expected_config:
            raise SystemExit(
                f"Refusing to resume {out_path}: existing config {prev_config} "
                f"does not match requested config {expected_config}."
            )
        trajectory = prev.get("trajectory", [])
        selected = [BENCH_IDS.index(row["added_benchmark"]) for row in trajectory]
        remaining = [j for j in candidates if j not in selected]

    cache_root = _cache_root(out_path, candidate_allowlist_ids)
    t_start = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(SEED,),
    ) as pool:
        for step in range(len(selected) + 1, args.max_steps + 1):
            if not remaining:
                break
            step_start = time.time()
            selected_ids = [BENCH_IDS[j] for j in selected]
            candidate_results: dict[str, dict[str, Any]] = {}
            futures = {}
            for cand_j in remaining:
                cache_path = probe_candidate_cache_path(cache_root, step, BENCH_IDS[cand_j])
                cached = _load_candidate_cache(cache_path, BENCH_IDS[cand_j], selected_ids)
                if cached is not None:
                    candidate_results[BENCH_IDS[cand_j]] = cached
                    continue
                futures[pool.submit(_evaluate_candidate, (selected + [cand_j], cand_j))] = cand_j

            for future in as_completed(futures):
                cand_j, predictions, score_metrics, ranking = future.result()
                record = _candidate_record(cand_j, predictions, score_metrics, ranking)
                write_json_atomic(
                    probe_candidate_cache_path(cache_root, step, BENCH_IDS[cand_j]),
                    {
                        "step": step,
                        "objective": OBJECTIVE,
                        "margin": MARGIN,
                        "eval_protocol": EVAL_PROTOCOL,
                        "probe_set_before_candidate": selected_ids,
                        "record": record,
                    },
                )
                candidate_results[BENCH_IDS[cand_j]] = record

            if len(candidate_results) != len(remaining):
                raise RuntimeError(
                    f"Step {step} has {len(candidate_results)} candidate results, "
                    f"expected {len(remaining)}"
                )

            best_j = None
            best_record = None
            best_score = float("inf")
            for cand_j in remaining:
                record = candidate_results[BENCH_IDS[cand_j]]
                score = float(record["score"])
                if score < best_score:
                    best_score = score
                    best_j = cand_j
                    best_record = record
            if best_j is None or best_record is None:
                raise RuntimeError(f"No finite candidate score at step {step}")

            selected.append(best_j)
            remaining.remove(best_j)
            trajectory.append({
                "step": step,
                "added_benchmark": BENCH_IDS[best_j],
                "added_benchmark_name": BENCH_NAMES.get(BENCH_IDS[best_j], BENCH_IDS[best_j]),
                "score": best_score,
                "pairwise_accuracy_margin5": best_record["pairwise_accuracy_margin5"],
                "n_pairs": best_record["n_pairs"],
                "n_correct": best_record["n_correct"],
                "n_predicted_ties": best_record["n_predicted_ties"],
                "probe_set": [BENCH_IDS[j] for j in selected],
                "elapsed_s": time.time() - step_start,
                "candidate_results": candidate_results,
            })

            output = {
                "config": {
                    "experiment": "sec5_findings/ranking_preservation/greedy_probe_set",
                    "objective": OBJECTIVE,
                    "margin": MARGIN,
                    "eval_protocol": EVAL_PROTOCOL,
                    "ranking_evaluation_universe": (
                        "all observed cells; probe cells are exact predictions "
                        "and remain in the pairwise denominator"
                    ),
                    "n_models": N_MODELS,
                    "n_bench": N_BENCH,
                    "n_observed": int(OBSERVED.sum()),
                    "n_candidates": len(candidates),
                    "candidate_allowlist_path": (
                        os.path.relpath(args.candidate_allowlist, REPO_ROOT)
                        if args.candidate_allowlist else None
                    ),
                    "candidate_allowlist_ids": candidate_allowlist_ids,
                    "candidate_limit": args.candidate_limit,
                    "max_steps": args.max_steps,
                    "seed": SEED,
                    "workers": args.workers,
                    "candidate_cache_dir": os.path.relpath(cache_root, SCRIPT_DIR),
                },
                "trajectory": trajectory,
            }
            write_json_atomic(out_path, output, indent=2)
            print(
                f"Step {step}: added {BENCH_IDS[best_j]} "
                f"acc={best_record['pairwise_accuracy_margin5']:.4f} "
                f"({time.time() - step_start:.1f}s)"
            )

    print(f"Saved {out_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
