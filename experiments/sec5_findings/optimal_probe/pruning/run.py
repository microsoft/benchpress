#!/usr/bin/env python3
"""Derive greedy-elimination pruning diagnostics from an existing greedy run.

The source greedy result must already contain candidate_results for each greedy
context. This script does not re-evaluate BenchPress predictions.
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import (  # noqa: E402
    BENCH_CATS,
    BENCH_IDS,
    BENCH_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    load_benchmark_allowlist,
)
from benchpress.io_utils import load_json, safe_token, write_json_atomic  # noqa: E402
from benchpress.shard_utils import short_text_hash  # noqa: E402

SEED = 42
EVAL_PROTOCOL = "all_known_probe_greedy_elimination_pruning_v1"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _parse_csv_ids(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_ids(value: str | None, label: str) -> list[str]:
    ids = _parse_csv_ids(value)
    unknown = [bid for bid in ids if bid not in BENCH_IDS]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    duplicates = sorted({bid for bid in ids if ids.count(bid) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}: {duplicates}")
    return ids


def _default_out(args: argparse.Namespace) -> str:
    fixed_ids = _resolve_ids(args.fixed_probes, "fixed probes")
    fixed_part = "none" if not fixed_ids else safe_token("-".join(fixed_ids))
    return os.path.join(
        RESULTS_DIR,
        f"greedy_elimination_{safe_token(args.metric)}"
        f"_fixed-{fixed_part}_from_existing_greedy.json.gz",
    )


def _candidate_ids_from_source(
    source_payload: dict, candidate_allowlist: str | None
) -> list[str]:
    if candidate_allowlist:
        _allowlist, allowlist_ids = load_benchmark_allowlist(
            candidate_allowlist, label="Candidate allowlist"
        )
        return list(allowlist_ids)

    config = source_payload.get("config", {})
    for key in ("candidate_allowlist_ids", "candidate_ids"):
        candidate_ids = config.get(key)
        if candidate_ids is not None:
            return list(candidate_ids)

    bench_ids = config.get("bench_ids")
    n_candidates = config.get("n_candidates")
    if bench_ids is not None and (
        n_candidates is None or int(n_candidates) == len(bench_ids)
    ):
        return list(bench_ids)
    if n_candidates is None or int(n_candidates) == len(BENCH_IDS):
        return list(BENCH_IDS)

    raise ValueError(
        "Cannot infer candidate IDs from source greedy result. Pass "
        "--candidate-allowlist matching the source result."
    )


def _gain_abs(baseline_score: float | None, candidate_score: float | None) -> float | None:
    if baseline_score is None or candidate_score is None:
        return None
    return float(baseline_score - candidate_score)


def _gain_rel(gain_abs: float | None, baseline_score: float | None) -> float | None:
    if gain_abs is None or baseline_score is None or baseline_score <= 0:
        return None
    return float(gain_abs / baseline_score)


def _unique_model_coverage(candidate_j: int, fixed_indices: list[int]) -> int:
    candidate_obs = OBSERVED[:, candidate_j]
    if not fixed_indices:
        return int(candidate_obs.sum())
    fixed_obs = OBSERVED[:, fixed_indices].any(axis=1)
    return int((candidate_obs & ~fixed_obs).sum())


def _threshold_pass(record: dict, max_gain_abs: float, max_gain_rel: float, mode: str) -> bool:
    gain_abs = record["max_gain_abs"]
    gain_rel = record["max_gain_rel"]
    abs_pass = gain_abs is not None and gain_abs <= max_gain_abs
    rel_pass = gain_rel is not None and gain_rel <= max_gain_rel
    if mode == "any":
        return abs_pass or rel_pass
    if mode == "all":
        return abs_pass and rel_pass
    raise ValueError(f"Unknown threshold mode: {mode}")


def _summarize_candidates(
    candidate_ids: list[str],
    fixed_ids: list[str],
    protected_ids: list[str],
    trajectory: list[dict],
    max_gain_abs: float,
    max_gain_rel: float,
    threshold_mode: str,
    max_unique_model_coverage: int,
    category_guard_top_n: int,
) -> dict:
    fixed_set = set(fixed_ids)
    protected_set = set(protected_ids)
    selected_ids = {step["added_benchmark"] for step in trajectory}
    fixed_indices = [BENCH_IDS.index(fid) for fid in fixed_ids]
    by_candidate = {
        bid: {
            "benchmark_id": bid,
            "benchmark_name": BENCH_NAMES.get(bid, bid),
            "benchmark_category": str(BENCH_CATS[BENCH_IDS.index(bid)]),
            "selected_by_greedy": bid in selected_ids,
            "fixed_probe": bid in fixed_set,
            "protected_probe": bid in protected_set,
            "unique_model_coverage_vs_fixed": _unique_model_coverage(
                BENCH_IDS.index(bid), fixed_indices
            ),
            "context_records": [],
        }
        for bid in candidate_ids
    }

    for step in trajectory:
        for bid, record in step["candidate_results"].items():
            if bid not in by_candidate:
                continue
            by_candidate[bid]["context_records"].append(
                {
                    "step": step["step"],
                    "selected_before": step["selected_before"],
                    "score": record["score"],
                    "medape": record["medape"],
                    "medae": record["medae"],
                    "gain_abs": record["gain_abs"],
                    "gain_rel": record["gain_rel"],
                }
            )

    for rec in by_candidate.values():
        gains_abs = [
            x["gain_abs"] for x in rec["context_records"] if x["gain_abs"] is not None
        ]
        gains_rel = [
            x["gain_rel"] for x in rec["context_records"] if x["gain_rel"] is not None
        ]
        rec["max_gain_abs"] = float(max(gains_abs)) if gains_abs else None
        rec["max_gain_rel"] = float(max(gains_rel)) if gains_rel else None
        rec["n_contexts_evaluated"] = len(rec["context_records"])

    category_top = set()
    if category_guard_top_n > 0:
        grouped: dict[str, list[dict]] = {}
        for rec in by_candidate.values():
            if rec["fixed_probe"] or rec["protected_probe"]:
                continue
            grouped.setdefault(rec["benchmark_category"], []).append(rec)
        for group in grouped.values():
            ranked = sorted(
                group,
                key=lambda r: (
                    -float("-inf" if r["max_gain_abs"] is None else r["max_gain_abs"]),
                    r["benchmark_id"],
                ),
            )
            for rec in ranked[: int(category_guard_top_n)]:
                category_top.add(rec["benchmark_id"])

    removable_ids = []
    kept_ids = []
    for bid in candidate_ids:
        rec = by_candidate[bid]
        guard_reasons = []
        if rec["fixed_probe"]:
            guard_reasons.append("fixed_probe")
        if rec["protected_probe"]:
            guard_reasons.append("protected_probe")
        if rec["selected_by_greedy"]:
            guard_reasons.append("selected_by_greedy")
        if rec["unique_model_coverage_vs_fixed"] > max_unique_model_coverage:
            guard_reasons.append("unique_model_coverage")
        if bid in category_top:
            guard_reasons.append("category_representative")
        if rec["n_contexts_evaluated"] == 0:
            guard_reasons.append("not_evaluated_in_any_context")

        threshold_pass = _threshold_pass(
            rec, max_gain_abs, max_gain_rel, threshold_mode
        )
        rec["threshold_pass"] = threshold_pass
        rec["guard_reasons"] = guard_reasons
        rec["decision"] = "remove" if threshold_pass and not guard_reasons else "keep"
        if rec["decision"] == "remove":
            removable_ids.append(bid)
        else:
            kept_ids.append(bid)

    return {
        "by_candidate": by_candidate,
        "removable_ids": removable_ids,
        "kept_ids": kept_ids,
    }


def _write_allowlist(path: str, payload: dict) -> None:
    result_rel = os.path.relpath(payload["result_path"], REPO_ROOT)
    allowlist_payload = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "description": (
            "Candidate allowlist generated by greedy-elimination pruning. "
            "This is a derived diagnostic artifact; see source_result for raw gains, "
            "guards, and thresholds."
        ),
        "source_result": result_rel,
        "pruning_config": payload["config"],
        "removed_benchmark_ids": payload["summary"]["removable_ids"],
        "benchmark_ids": payload["summary"]["kept_ids"],
    }
    write_json_atomic(path, allowlist_payload, indent=2, trailing_newline=True)


def _source_baseline_record(source_steps: list[dict], source_step_index: int) -> dict:
    if source_step_index <= 0:
        raise ValueError(
            "Source-greedy reuse needs a non-empty fixed probe prefix so the "
            "baseline context is present in the source greedy trajectory."
        )
    prev = source_steps[source_step_index - 1]
    added = prev["added_benchmark"]
    record = prev.get("candidate_results", {}).get(added)
    if record is None:
        raise ValueError(f"Source result is missing selected record for {added}")
    return record


def _derive_trajectory(
    source_payload: dict,
    fixed_ids: list[str],
    max_steps: int,
    metric: str,
) -> list[dict]:
    source_config = source_payload.get("config", {})
    source_metric = source_config.get("metric")
    if source_metric != metric:
        raise ValueError(f"Source greedy metric is {source_metric!r}, expected {metric!r}")

    source_steps = list(source_payload.get("trajectory", []))
    required_steps = len(fixed_ids) + max_steps
    if len(source_steps) < required_steps:
        raise ValueError(
            f"Source greedy result has {len(source_steps)} steps, but "
            f"{required_steps} are required."
        )
    source_prefix = [step["added_benchmark"] for step in source_steps[: len(fixed_ids)]]
    if source_prefix != fixed_ids:
        raise ValueError(
            f"Source greedy prefix {source_prefix} does not match fixed probes {fixed_ids}"
        )

    trajectory = []
    for local_step in range(1, max_steps + 1):
        source_step_index = len(fixed_ids) + local_step - 1
        source_step = source_steps[source_step_index]
        baseline = _source_baseline_record(source_steps, source_step_index)
        baseline_score = baseline["score"]

        candidate_results = {}
        for bid, source_record in source_step.get("candidate_results", {}).items():
            if bid not in BENCH_IDS:
                raise ValueError(f"Unknown benchmark in source result: {bid}")
            record = dict(source_record)
            gain_abs = _gain_abs(baseline_score, record["score"])
            record["gain_abs"] = gain_abs
            record["gain_rel"] = _gain_rel(gain_abs, baseline_score)
            record.setdefault("benchmark_id", bid)
            record.setdefault("benchmark_name", BENCH_NAMES.get(bid, bid))
            record.setdefault(
                "benchmark_category", str(BENCH_CATS[BENCH_IDS.index(bid)])
            )
            candidate_results[bid] = record

        added_score = source_step["score"]
        added_gain = _gain_abs(baseline_score, added_score)
        trajectory.append(
            {
                "step": local_step,
                "source_greedy_step": source_step["step"],
                "selected_before": list(source_step["probe_set"][:-1]),
                "baseline": baseline,
                "added_benchmark": source_step["added_benchmark"],
                "added_benchmark_name": source_step["added_benchmark_name"],
                "score": added_score,
                "medape": source_step["medape"],
                "medae": source_step["medae"],
                "gain_abs": added_gain,
                "gain_rel": _gain_rel(added_gain, baseline_score),
                "probe_set": source_step["probe_set"],
                "candidate_results": candidate_results,
            }
        )
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-greedy-result", required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--metric", choices=["medape", "medae"], default="medae")
    parser.add_argument("--fixed-probes", default="gpqa_diamond")
    parser.add_argument("--protected-probes", default="gpqa_diamond,mmlu_pro")
    parser.add_argument("--candidate-allowlist", default=None)
    parser.add_argument("--max-gain-abs", type=float, default=0.05)
    parser.add_argument("--max-gain-rel", type=float, default=0.01)
    parser.add_argument("--threshold-mode", choices=["any", "all"], default="any")
    parser.add_argument("--max-unique-model-coverage", type=int, default=0)
    parser.add_argument("--category-guard-top-n", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allowlist-out", default=None)
    args = parser.parse_args()

    source_greedy_result = os.path.abspath(args.source_greedy_result)
    source_payload = load_json(source_greedy_result)
    candidate_ids = _candidate_ids_from_source(source_payload, args.candidate_allowlist)
    unknown_candidates = [bid for bid in candidate_ids if bid not in BENCH_IDS]
    if unknown_candidates:
        raise ValueError(f"Unknown candidates: {unknown_candidates}")

    fixed_ids = _resolve_ids(args.fixed_probes, "fixed probes")
    protected_ids = _resolve_ids(args.protected_probes, "protected probes")
    missing_fixed = [bid for bid in fixed_ids if bid not in candidate_ids]
    if missing_fixed:
        raise ValueError(f"Fixed probes must be in candidate set: {missing_fixed}")

    out_path = os.path.abspath(args.out) if args.out else _default_out(args)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    config = {
        "eval_protocol": EVAL_PROTOCOL,
        "metric": args.metric,
        "seed": SEED,
        "n_models": N_MODELS,
        "n_bench": N_BENCH,
        "n_observed": int(OBSERVED.sum()),
        "candidate_allowlist_path": (
            os.path.relpath(args.candidate_allowlist, REPO_ROOT)
            if args.candidate_allowlist
            else None
        ),
        "candidate_ids": candidate_ids,
        "candidate_hash": short_text_hash("\n".join(candidate_ids), n=12),
        "fixed_probe_ids": fixed_ids,
        "fixed_probe_hash": short_text_hash("\n".join(fixed_ids), n=12),
        "protected_probe_ids": protected_ids,
        "protected_probe_hash": short_text_hash("\n".join(protected_ids), n=12),
        "max_steps": int(args.max_steps),
        "max_gain_abs": float(args.max_gain_abs),
        "max_gain_rel": float(args.max_gain_rel),
        "threshold_mode": args.threshold_mode,
        "max_unique_model_coverage": int(args.max_unique_model_coverage),
        "category_guard_top_n": int(args.category_guard_top_n),
        "source_greedy_result_path": os.path.relpath(source_greedy_result, REPO_ROOT),
    }

    trajectory = _derive_trajectory(
        source_payload,
        fixed_ids,
        int(args.max_steps),
        args.metric,
    )
    summary = _summarize_candidates(
        candidate_ids,
        fixed_ids,
        protected_ids,
        trajectory,
        config["max_gain_abs"],
        config["max_gain_rel"],
        config["threshold_mode"],
        config["max_unique_model_coverage"],
        config["category_guard_top_n"],
    )
    output = {
        "config": config,
        "trajectory": trajectory,
        "summary": summary,
        "result_path": out_path,
        "source_greedy_result": config["source_greedy_result_path"],
    }
    write_json_atomic(out_path, output, indent=2)

    if args.allowlist_out:
        allowlist_out = os.path.abspath(args.allowlist_out)
        _write_allowlist(allowlist_out, output)
        output["allowlist_out"] = os.path.relpath(allowlist_out, REPO_ROOT)
        write_json_atomic(out_path, output, indent=2)
        print(f"Allowlist saved -> {allowlist_out}")

    print(f"Saved -> {out_path}")
    print(f"Removed {len(summary['removable_ids'])}: {summary['removable_ids']}")


if __name__ == "__main__":
    main()
