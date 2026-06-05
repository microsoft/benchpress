#!/usr/bin/env python3
"""Greedy-elimination pruning for all-known probe candidates.

The script uses greedy selection to create realistic contexts, then records the
conditional gain of every unselected candidate in those contexts. The output is
a pruning diagnostic: benchmarks are removed only by explicit threshold and
guard rules, not because greedy did or did not select them.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
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

from benchpress.all_methods import predict_benchpress_scores
from benchpress.evaluation_harness import (
    BENCH_CATS,
    BENCH_IDS,
    BENCH_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    evaluate_probe_set,
    load_benchmark_allowlist,
    pack_probe_predictions,
    probe_candidate_cache_path,
)
from benchpress.io_utils import load_json, safe_token, write_json_atomic
from benchpress.shard_utils import short_text_hash

SEED = 42
EVAL_PROTOCOL = "all_known_probe_greedy_elimination_pruning_v1"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

np.random.seed(SEED)
random.seed(SEED)


def _init_worker(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _parse_csv_ids(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _candidate_source_label(path: str | None) -> str:
    if path is None:
        return "all"
    name = os.path.basename(path)
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            return safe_token(name[: -len(suffix)])
    return safe_token(name)


def _load_candidates(candidate_allowlist: str | None, candidate_limit: int | None):
    allowlist, allowlist_ids = load_benchmark_allowlist(
        candidate_allowlist, label="Candidate allowlist"
    )
    if allowlist is None:
        candidate_ids = list(BENCH_IDS)
    else:
        candidate_ids = list(allowlist_ids)
    if candidate_limit is not None:
        candidate_ids = candidate_ids[: int(candidate_limit)]
    if not candidate_ids:
        raise ValueError("Candidate set is empty")
    return [BENCH_IDS.index(bid) for bid in candidate_ids], candidate_ids


def _resolve_ids(value: str | None, label: str) -> list[str]:
    ids = _parse_csv_ids(value)
    unknown = [bid for bid in ids if bid not in BENCH_IDS]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    duplicates = sorted({bid for bid in ids if ids.count(bid) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}: {duplicates}")
    return ids


def _default_out(args, candidate_label: str) -> str:
    fixed_ids = _resolve_ids(args.fixed_probes, "fixed probes")
    fixed_part = "none" if not fixed_ids else safe_token("-".join(fixed_ids))
    limit_part = "" if args.candidate_limit is None else f"_limit{int(args.candidate_limit)}"
    return os.path.join(
        RESULTS_DIR,
        (
            f"greedy_elimination_{safe_token(args.metric)}"
            f"_fixed-{fixed_part}_candidates-{candidate_label}{limit_part}.json.gz"
        ),
    )


def _cache_root(out_path: str, config: dict) -> str:
    stem = os.path.basename(out_path)
    if stem.endswith(".json.gz"):
        stem = stem[: -len(".json.gz")]
    else:
        stem = os.path.splitext(stem)[0]
    keys = [
        EVAL_PROTOCOL,
        config["metric"],
        "\n".join(config["candidate_ids"]),
        "\n".join(config["fixed_probe_ids"]),
        "\n".join(config["protected_probe_ids"]),
        str(config["max_steps"]),
        str(config["max_gain_abs"]),
        str(config["max_gain_rel"]),
        config["threshold_mode"],
        str(config["max_unique_model_coverage"]),
        str(config["category_guard_top_n"]),
    ]
    digest = short_text_hash("\n---\n".join(keys), n=12)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if os.path.commonpath([out_dir, os.path.abspath(RESULTS_DIR)]) == os.path.abspath(
        RESULTS_DIR
    ):
        cache_parent = os.path.join(RESULTS_DIR, ".candidate_cache")
    else:
        cache_parent = os.path.join(out_dir, ".candidate_cache")
    return os.path.join(cache_parent, f"{safe_token(stem)}__{digest}")


def _eval_probe_set(probe_indices: list[int], metric: str):
    predictions, metrics, score = evaluate_probe_set(
        probe_indices,
        predict_benchpress_scores,
        metric=metric,
    )
    return {
        "probe_set": [BENCH_IDS[j] for j in probe_indices],
        "probe_names": [BENCH_NAMES.get(BENCH_IDS[j], BENCH_IDS[j]) for j in probe_indices],
        "score": float(score) if np.isfinite(score) else None,
        "medape": float(metrics["medape"]) if np.isfinite(metrics["medape"]) else None,
        "medae": float(metrics["medae"]) if np.isfinite(metrics["medae"]) else None,
        "n": int(metrics["n"]),
        "predictions": pack_probe_predictions(predictions),
    }


def _eval_candidate(job):
    selected, cand_j, metric = job
    t0 = time.time()
    record = _eval_probe_set(selected + [cand_j], metric)
    record.update(
        {
            "benchmark_id": BENCH_IDS[cand_j],
            "benchmark_name": BENCH_NAMES.get(BENCH_IDS[cand_j], BENCH_IDS[cand_j]),
            "benchmark_category": str(BENCH_CATS[cand_j]),
            "elapsed_s": time.time() - t0,
        }
    )
    return cand_j, record


def _cache_payload_config(config: dict) -> dict:
    keys = [
        "eval_protocol",
        "metric",
        "n_models",
        "n_bench",
        "n_observed",
        "candidate_hash",
        "fixed_probe_hash",
        "protected_probe_hash",
        "max_steps",
        "max_gain_abs",
        "max_gain_rel",
        "threshold_mode",
        "max_unique_model_coverage",
        "category_guard_top_n",
    ]
    return {key: config[key] for key in keys}


def _load_candidate_cache(path: str, config: dict, expected_bid: str, selected_ids: list[str]):
    if not os.path.exists(path):
        return None
    payload = load_json(path)
    if payload.get("config") != _cache_payload_config(config):
        raise RuntimeError(
            f"Candidate cache config mismatch in {path}. Use a different --out "
            "or delete the incompatible cache."
        )
    if payload.get("selected_probe_ids") != selected_ids:
        raise RuntimeError(
            f"Candidate cache context mismatch in {path}: expected {selected_ids}, "
            f"found {payload.get('selected_probe_ids')}"
        )
    record = payload.get("record", {})
    if record.get("benchmark_id") != expected_bid:
        raise RuntimeError(
            f"Candidate cache benchmark mismatch in {path}: expected {expected_bid}, "
            f"found {record.get('benchmark_id')}"
        )
    return record


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
):
    fixed_set = set(fixed_ids)
    protected_set = set(protected_ids)
    selected_ids = {step["added_benchmark"] for step in trajectory}
    by_candidate = {
        bid: {
            "benchmark_id": bid,
            "benchmark_name": BENCH_NAMES.get(bid, bid),
            "benchmark_category": str(BENCH_CATS[BENCH_IDS.index(bid)]),
            "selected_by_greedy": bid in selected_ids,
            "fixed_probe": bid in fixed_set,
            "protected_probe": bid in protected_set,
            "unique_model_coverage_vs_fixed": _unique_model_coverage(
                BENCH_IDS.index(bid), [BENCH_IDS.index(fid) for fid in fixed_ids]
            ),
            "context_records": [],
        }
        for bid in candidate_ids
    }

    for step in trajectory:
        for bid, record in step["candidate_results"].items():
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--metric", choices=["medape", "medae"], default="medae")
    parser.add_argument("--fixed-probes", default="gpqa_diamond")
    parser.add_argument("--protected-probes", default="gpqa_diamond,mmlu_pro")
    parser.add_argument("--candidate-allowlist", default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--max-gain-abs", type=float, default=0.05)
    parser.add_argument("--max-gain-rel", type=float, default=0.01)
    parser.add_argument("--threshold-mode", choices=["any", "all"], default="any")
    parser.add_argument("--max-unique-model-coverage", type=int, default=0)
    parser.add_argument("--category-guard-top-n", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allowlist-out", default=None)
    args = parser.parse_args()

    candidate_indices, candidate_ids = _load_candidates(
        args.candidate_allowlist, args.candidate_limit
    )
    fixed_ids = _resolve_ids(args.fixed_probes, "fixed probes")
    protected_ids = _resolve_ids(args.protected_probes, "protected probes")
    if args.candidate_limit is not None:
        for bid in fixed_ids + protected_ids:
            if bid not in candidate_ids:
                candidate_ids.append(bid)
        candidate_indices = [BENCH_IDS.index(bid) for bid in candidate_ids]
    missing_fixed = [bid for bid in fixed_ids if bid not in candidate_ids]
    if missing_fixed:
        raise ValueError(f"Fixed probes must be in candidate set: {missing_fixed}")

    fixed_indices = [BENCH_IDS.index(bid) for bid in fixed_ids]
    selected = list(fixed_indices)
    remaining = [j for j in candidate_indices if j not in set(selected)]
    candidate_label = _candidate_source_label(args.candidate_allowlist)
    out_path = args.out or _default_out(args, candidate_label)
    if not os.path.isabs(out_path):
        out_path = os.path.join(RESULTS_DIR, out_path)
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
        "candidate_limit": args.candidate_limit,
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
        "workers": int(args.workers),
        "prediction_engine": "predict_benchpress_scores (Logit Bias ALS, rank=2, lambda=0.1)",
        "cell_masking": (
            "For model i, keep probe cells visible and mask non-probe cells. "
            "Probe target cells are known and stored with pred=true; non-probe "
            "target cells are predicted by BenchPress. The evaluation universe "
            "is fixed to all observed cells."
        ),
    }
    cache_root = _cache_root(out_path, config)
    config["candidate_cache_dir"] = os.path.relpath(cache_root, SCRIPT_DIR)

    trajectory = []
    if os.path.exists(out_path):
        prev = load_json(out_path)
        prev_config = prev.get("config", {})
        expected = {k: config[k] for k in _cache_payload_config(config)}
        found = {k: prev_config.get(k) for k in expected}
        if found != expected:
            raise SystemExit(
                f"Refusing to resume {out_path}: existing config {found} does "
                f"not match requested config {expected}."
            )
        trajectory = prev.get("trajectory", [])
        selected = [BENCH_IDS.index(bid) for bid in fixed_ids]
        for step in trajectory:
            selected.append(BENCH_IDS.index(step["added_benchmark"]))
        remaining = [j for j in candidate_indices if j not in set(selected)]
        print(f"Resuming {out_path}: {len(trajectory)} completed greedy steps")

    print(f"Matrix: {N_MODELS}x{N_BENCH}, observed={int(OBSERVED.sum())}")
    print(f"Metric: {args.metric}")
    print(f"Fixed probes: {fixed_ids}")
    print(f"Protected probes: {protected_ids}")
    print(f"Candidates: {len(candidate_ids)}")
    print(f"Workers: {args.workers}")

    t_all = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(SEED,),
    ) as pool:
        while len(trajectory) < args.max_steps and remaining:
            step = len(trajectory) + 1
            selected_ids = [BENCH_IDS[j] for j in selected]
            print(f"\n--- Context {step}/{args.max_steps}: {selected_ids} ---")
            step_t0 = time.time()
            baseline = _eval_probe_set(selected, args.metric)
            baseline_score = baseline["score"]

            candidate_results = {}
            futures = {}
            for cand_j in remaining:
                bid = BENCH_IDS[cand_j]
                cache_path = probe_candidate_cache_path(cache_root, step, bid)
                cached = _load_candidate_cache(cache_path, config, bid, selected_ids)
                if cached is not None:
                    candidate_results[bid] = cached
                    continue
                futures[pool.submit(_eval_candidate, (selected, cand_j, args.metric))] = cand_j

            for future in as_completed(futures):
                cand_j, record = future.result()
                bid = BENCH_IDS[cand_j]
                gain_abs = _gain_abs(baseline_score, record["score"])
                gain_rel = _gain_rel(gain_abs, baseline_score)
                record["gain_abs"] = gain_abs
                record["gain_rel"] = gain_rel
                write_json_atomic(
                    probe_candidate_cache_path(cache_root, step, bid),
                    {
                        "config": _cache_payload_config(config),
                        "step": step,
                        "selected_probe_ids": selected_ids,
                        "baseline_score": baseline_score,
                        "record": record,
                    },
                )
                candidate_results[bid] = record

            for bid, record in list(candidate_results.items()):
                if "gain_abs" not in record:
                    gain_abs = _gain_abs(baseline_score, record["score"])
                    record["gain_abs"] = gain_abs
                    record["gain_rel"] = _gain_rel(gain_abs, baseline_score)
                    candidate_results[bid] = record

            if len(candidate_results) != len(remaining):
                raise RuntimeError(
                    f"Step {step} has {len(candidate_results)} candidate results, "
                    f"expected {len(remaining)}"
                )

            best_j = min(
                remaining,
                key=lambda j: (
                    float("inf")
                    if candidate_results[BENCH_IDS[j]]["score"] is None
                    else candidate_results[BENCH_IDS[j]]["score"]
                ),
            )
            best_record = candidate_results[BENCH_IDS[best_j]]
            selected.append(best_j)
            remaining.remove(best_j)
            step_record = {
                "step": step,
                "selected_before": selected_ids,
                "baseline": baseline,
                "added_benchmark": BENCH_IDS[best_j],
                "added_benchmark_name": BENCH_NAMES.get(BENCH_IDS[best_j], BENCH_IDS[best_j]),
                "score": best_record["score"],
                "medape": best_record["medape"],
                "medae": best_record["medae"],
                "gain_abs": best_record["gain_abs"],
                "gain_rel": best_record["gain_rel"],
                "probe_set": [BENCH_IDS[j] for j in selected],
                "elapsed_s": time.time() - step_t0,
                "candidate_results": candidate_results,
            }
            trajectory.append(step_record)
            summary = _summarize_candidates(
                candidate_ids,
                fixed_ids,
                protected_ids,
                trajectory,
                args.max_gain_abs,
                args.max_gain_rel,
                args.threshold_mode,
                args.max_unique_model_coverage,
                args.category_guard_top_n,
            )
            output = {
                "config": config,
                "trajectory": trajectory,
                "summary": summary,
                "result_path": out_path,
                "elapsed_s": time.time() - t_all,
            }
            write_json_atomic(out_path, output, indent=2)
            print(
                f"  Added {BENCH_IDS[best_j]:30s} "
                f"score={best_record['score']:.4f} "
                f"gain={best_record['gain_abs']:.4f} "
                f"removable_so_far={len(summary['removable_ids'])} "
                f"[{time.time() - step_t0:.1f}s]"
            )

    final = load_json(out_path)
    if args.allowlist_out:
        allowlist_out = args.allowlist_out
        if not os.path.isabs(allowlist_out):
            allowlist_out = os.path.abspath(os.path.join(SCRIPT_DIR, allowlist_out))
        _write_allowlist(allowlist_out, final)
        final["allowlist_out"] = os.path.relpath(allowlist_out, REPO_ROOT)
        write_json_atomic(out_path, final, indent=2)
        print(f"Allowlist saved -> {allowlist_out}")

    print(f"\nSaved -> {out_path}")
    print(f"Total time: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
