#!/usr/bin/env python3
"""Rank-prune benchmark candidates from an existing all-known greedy run."""

from __future__ import annotations

import argparse
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import BENCH_CATS, BENCH_IDS, BENCH_NAMES  # noqa: E402
from benchpress.io_utils import load_json, safe_token, write_json_atomic  # noqa: E402
from benchpress.shard_utils import short_text_hash  # noqa: E402

EVAL_PROTOCOL = "all_known_greedy_rank_pruning_v1"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _default_out(metric: str, keep_fraction: float) -> str:
    keep_pct = int(round(keep_fraction * 100))
    return os.path.join(
        RESULTS_DIR,
        f"rank_pruning_{safe_token(metric)}_top{keep_pct}_from_existing_greedy.json",
    )


def _candidate_ids_from_source(source_payload: dict) -> list[str]:
    config = source_payload.get("config", {})
    for key in ("candidate_allowlist_ids", "candidate_ids", "bench_ids"):
        candidate_ids = config.get(key)
        if candidate_ids is not None:
            return list(candidate_ids)
    return list(BENCH_IDS)


def _rank_step(source_step: dict) -> list[dict]:
    records = []
    for bid, record in source_step.get("candidate_results", {}).items():
        if bid not in BENCH_IDS:
            raise ValueError(f"Unknown benchmark in source result: {bid}")
        score = record.get("score")
        records.append((bid, float("inf") if score is None else float(score)))
    records.sort(key=lambda item: (item[1], item[0]))

    denom = max(1, len(records) - 1)
    ranked = []
    for rank, (bid, score) in enumerate(records, start=1):
        ranked.append(
            {
                "benchmark_id": bid,
                "score": score if math.isfinite(score) else None,
                "rank": rank,
                "n_candidates": len(records),
                "rank_percentile": float((rank - 1) / denom),
            }
        )
    return ranked


def _rank_prune(
    source_payload: dict,
    metric: str,
    keep_fraction: float,
    max_steps: int | None,
) -> dict:
    source_metric = source_payload.get("config", {}).get("metric")
    if source_metric != metric:
        raise ValueError(f"Source greedy metric is {source_metric!r}, expected {metric!r}")

    candidate_ids = _candidate_ids_from_source(source_payload)
    unknown = [bid for bid in candidate_ids if bid not in BENCH_IDS]
    if unknown:
        raise ValueError(f"Unknown candidate IDs in source result: {unknown}")

    source_steps = list(source_payload.get("trajectory", []))
    if not source_steps:
        raise ValueError("Source greedy result has no trajectory")
    if max_steps is None:
        used_steps = source_steps
    else:
        if max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        if len(source_steps) < max_steps:
            raise ValueError(
                f"Source greedy result has {len(source_steps)} steps, "
                f"but {max_steps} were requested."
            )
        used_steps = source_steps[:max_steps]

    by_candidate = {
        bid: {
            "benchmark_id": bid,
            "benchmark_name": BENCH_NAMES.get(bid, bid),
            "benchmark_category": str(BENCH_CATS[BENCH_IDS.index(bid)]),
            "selected_by_greedy_step": None,
            "rank_records": [],
        }
        for bid in candidate_ids
    }
    ranked_steps = []
    for source_step in used_steps:
        source_step_number = int(source_step["step"])
        added = source_step["added_benchmark"]
        if added in by_candidate:
            by_candidate[added]["selected_by_greedy_step"] = source_step_number

        ranked = _rank_step(source_step)
        ranked_steps.append(
            {
                "source_greedy_step": source_step_number,
                "added_benchmark": added,
                "added_benchmark_name": source_step.get(
                    "added_benchmark_name", BENCH_NAMES.get(added, added)
                ),
                "n_candidates": len(ranked),
                "ranks": ranked,
            }
        )
        for record in ranked:
            bid = record["benchmark_id"]
            if bid not in by_candidate:
                continue
            by_candidate[bid]["rank_records"].append(
                {
                    "source_greedy_step": source_step_number,
                    "rank": record["rank"],
                    "n_candidates": record["n_candidates"],
                    "rank_percentile": record["rank_percentile"],
                    "score": record["score"],
                }
            )

    for rec in by_candidate.values():
        rank_records = rec["rank_records"]
        rec["n_ranked_steps"] = len(rank_records)
        rec["avg_rank_percentile"] = (
            float(
                sum(item["rank_percentile"] for item in rank_records)
                / len(rank_records)
            )
            if rank_records
            else 1.0
        )
        rec["avg_rank"] = (
            float(sum(item["rank"] for item in rank_records) / len(rank_records))
            if rank_records
            else None
        )

    keep_count = math.ceil(len(candidate_ids) * keep_fraction)
    ranked_candidates = sorted(
        candidate_ids,
        key=lambda bid: (
            by_candidate[bid]["avg_rank_percentile"],
            by_candidate[bid]["avg_rank"]
            if by_candidate[bid]["avg_rank"] is not None
            else float("inf"),
            bid,
        ),
    )
    kept_ids = ranked_candidates[:keep_count]
    removed_ids = ranked_candidates[keep_count:]
    kept_set = set(kept_ids)
    for rank, bid in enumerate(ranked_candidates, start=1):
        rec = by_candidate[bid]
        rec["aggregate_rank"] = rank
        rec["decision"] = "keep" if bid in kept_set else "remove"

    return {
        "candidate_ids": candidate_ids,
        "ranked_steps": ranked_steps,
        "by_candidate": by_candidate,
        "kept_ids": kept_ids,
        "removed_ids": removed_ids,
        "keep_count": keep_count,
    }


def _write_allowlist(path: str, payload: dict) -> None:
    result_rel = os.path.relpath(payload["result_path"], REPO_ROOT)
    allowlist_payload = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "description": (
            "Candidate allowlist generated by greedy rank pruning. "
            "Benchmarks are kept by aggregate normalized rank across source "
            "greedy steps; see source_result and pruning_config."
        ),
        "source_result": result_rel,
        "source_greedy_result": payload["source_greedy_result"],
        "pruning_config": payload["config"],
        "removed_benchmark_ids": payload["summary"]["removed_ids"],
        "benchmark_ids": payload["summary"]["kept_ids"],
    }
    write_json_atomic(path, allowlist_payload, indent=2, trailing_newline=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-greedy-result", required=True)
    parser.add_argument("--metric", choices=["medape", "medae"], default="medae")
    parser.add_argument("--keep-fraction", type=float, default=0.30)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allowlist-out", default=None)
    args = parser.parse_args()

    if args.keep_fraction <= 0 or args.keep_fraction >= 1:
        raise ValueError("--keep-fraction must be between 0 and 1")

    source_greedy_result = os.path.abspath(args.source_greedy_result)
    source_payload = load_json(source_greedy_result)
    out_path = (
        os.path.abspath(args.out)
        if args.out
        else _default_out(args.metric, args.keep_fraction)
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = _rank_prune(
        source_payload,
        args.metric,
        args.keep_fraction,
        args.max_steps,
    )
    config = {
        "eval_protocol": EVAL_PROTOCOL,
        "metric": args.metric,
        "keep_fraction": float(args.keep_fraction),
        "keep_count": summary["keep_count"],
        "source_steps_used": (
            args.max_steps
            if args.max_steps is not None
            else len(source_payload.get("trajectory", []))
        ),
        "candidate_ids": summary["candidate_ids"],
        "candidate_hash": short_text_hash("\n".join(summary["candidate_ids"]), n=12),
        "source_greedy_result_path": os.path.relpath(source_greedy_result, REPO_ROOT),
        "rank_direction": "lower score is better within each greedy context",
        "missing_after_selection": (
            "Candidates selected by greedy are ranked only until their selected step; "
            "later contexts omit already-selected candidates."
        ),
    }
    output = {
        "config": config,
        "summary": summary,
        "result_path": out_path,
        "source_greedy_result": config["source_greedy_result_path"],
    }
    write_json_atomic(out_path, output, indent=2, trailing_newline=True)

    if args.allowlist_out:
        allowlist_out = os.path.abspath(args.allowlist_out)
        _write_allowlist(allowlist_out, output)
        output["allowlist_out"] = os.path.relpath(allowlist_out, REPO_ROOT)
        write_json_atomic(out_path, output, indent=2, trailing_newline=True)
        print(f"Allowlist saved -> {allowlist_out}")

    print(f"Saved -> {out_path}")
    print(f"Kept {len(summary['kept_ids'])}: {summary['kept_ids']}")
    print(f"Removed {len(summary['removed_ids'])}: {summary['removed_ids']}")


if __name__ == "__main__":
    main()
