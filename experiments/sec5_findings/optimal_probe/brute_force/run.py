#!/usr/bin/env python
"""Exhaustive probe-set search for the all-known BenchPress protocol."""

from __future__ import annotations

import argparse
import itertools
import math
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
    BENCH_IDS,
    BENCH_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    evaluate_probe_set,
    load_benchmark_allowlist,
    pack_probe_predictions,
)
from benchpress.io_utils import load_json, safe_token, write_json_atomic
from benchpress.shard_utils import short_text_hash

SEED = 42
EVAL_PROTOCOL = "all_known_probe_bruteforce_v1"
DEFAULT_CHUNK_SIZE = 256

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
            name = name[: -len(suffix)]
            break
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
    candidate_indices = [BENCH_IDS.index(bid) for bid in candidate_ids]
    if not candidate_indices:
        raise ValueError("Candidate set is empty")
    return candidate_indices, candidate_ids


def _resolve_fixed_probe_ids(fixed_probe: str | None) -> list[str]:
    fixed_ids = _parse_csv_ids(fixed_probe)
    unknown = [bid for bid in fixed_ids if bid not in BENCH_IDS]
    if unknown:
        raise ValueError(f"Unknown fixed probe IDs: {unknown}")
    duplicates = sorted({bid for bid in fixed_ids if fixed_ids.count(bid) > 1})
    if duplicates:
        raise ValueError(f"Duplicate fixed probe IDs: {duplicates}")
    return fixed_ids


def _assignment_residue(wave_index: int, num_waves: int, shard_index: int, num_shards: int):
    return int(wave_index) + int(num_waves) * int(shard_index)


def _assigned_count(total: int, residue: int, modulus: int) -> int:
    if residue >= total:
        return 0
    return ((total - 1 - residue) // modulus) + 1


def _default_out_dir(args, candidate_label: str) -> str:
    fixed = _resolve_fixed_probe_ids(args.fixed_probe)
    fixed_part = "" if not fixed else "_fixed-" + safe_token("-".join(fixed))
    limit_part = "" if args.candidate_limit is None else f"_limit{int(args.candidate_limit)}"
    return os.path.join(
        SCRIPT_DIR,
        "results",
        f"exhaustive_{safe_token(args.metric)}_k{int(args.k)}_candidates-{candidate_label}{limit_part}{fixed_part}",
    )


def _build_config(args, candidates: list[int], candidate_ids: list[str]):
    fixed_ids = _resolve_fixed_probe_ids(args.fixed_probe)
    fixed_indices = [BENCH_IDS.index(bid) for bid in fixed_ids]
    remaining_candidates = [j for j in candidates if j not in set(fixed_indices)]
    remaining_candidate_ids = [BENCH_IDS[j] for j in remaining_candidates]
    choose_size = int(args.k) - len(fixed_indices)
    if choose_size < 0:
        raise ValueError(
            f"k={args.k} is smaller than number of fixed probes ({len(fixed_indices)})"
        )
    if choose_size > len(remaining_candidates):
        raise ValueError(
            f"Cannot choose {choose_size} additional probes from "
            f"{len(remaining_candidates)} candidates"
        )

    total_combinations = math.comb(len(remaining_candidates), choose_size)
    num_waves = int(args.num_waves)
    num_shards = int(args.num_shards)
    wave_index = int(args.wave_index)
    shard_index = int(args.shard_index)
    if not (0 <= wave_index < num_waves):
        raise ValueError(f"wave_index must be in [0, {num_waves}), got {wave_index}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    residue = _assignment_residue(wave_index, num_waves, shard_index, num_shards)
    modulus = num_waves * num_shards

    candidate_hash = short_text_hash("\n".join(candidate_ids), n=12)
    remaining_hash = short_text_hash("\n".join(remaining_candidate_ids), n=12)
    fixed_hash = short_text_hash("\n".join(fixed_ids), n=12) if fixed_ids else None

    return {
        "eval_protocol": EVAL_PROTOCOL,
        "k": int(args.k),
        "metric": args.metric,
        "seed": SEED,
        "n_models": N_MODELS,
        "n_bench": N_BENCH,
        "n_observed": int(OBSERVED.sum()),
        "n_target_cells": int(OBSERVED.sum()),
        "eval_scope": "all_observed_cells",
        "prediction_engine": "predict_benchpress_scores (Logit Bias ALS, rank=2, lambda=0.1)",
        "candidate_allowlist_path": (
            os.path.relpath(args.candidate_allowlist, REPO_ROOT)
            if args.candidate_allowlist
            else None
        ),
        "candidate_limit": args.candidate_limit,
        "candidate_ids": candidate_ids,
        "candidate_hash": candidate_hash,
        "fixed_probe_ids": fixed_ids,
        "fixed_probe_hash": fixed_hash,
        "remaining_candidate_ids": remaining_candidate_ids,
        "remaining_candidate_hash": remaining_hash,
        "choose_size_after_fixed": choose_size,
        "total_combinations": total_combinations,
        "num_waves": num_waves,
        "num_shards": num_shards,
        "wave_index": wave_index,
        "shard_index": shard_index,
        "assignment_residue": residue,
        "assignment_modulus": modulus,
        "assigned_combinations": _assigned_count(total_combinations, residue, modulus),
        "chunk_size": int(args.chunk_size),
        "cell_masking": (
            "For model i, keep probe cells visible and mask non-probe cells. "
            "Probe target cells are known and stored with pred=true; non-probe "
            "target cells are predicted by BenchPress. The evaluation universe "
            "is fixed to all observed cells."
        ),
    }, fixed_indices, remaining_candidates


def _config_for_chunk(config: dict) -> dict:
    keys = [
        "eval_protocol",
        "k",
        "metric",
        "seed",
        "n_models",
        "n_bench",
        "n_observed",
        "n_target_cells",
        "candidate_hash",
        "fixed_probe_hash",
        "remaining_candidate_hash",
        "choose_size_after_fixed",
        "total_combinations",
        "num_waves",
        "num_shards",
        "wave_index",
        "shard_index",
        "assignment_residue",
        "assignment_modulus",
        "chunk_size",
    ]
    return {key: config.get(key) for key in keys}


def _chunk_path(out_dir: str, wave_index: int, shard_index: int, chunk_index: int) -> str:
    return os.path.join(
        out_dir,
        "shards",
        f"wave_{int(wave_index):02d}",
        f"shard_{int(shard_index):03d}",
        f"chunk_{int(chunk_index):06d}.json.gz",
    )


def _is_valid_existing_chunk(path: str, config: dict, combo_indices: list[int]) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) <= 100:
        return False
    payload = load_json(path)
    if payload.get("config") != _config_for_chunk(config):
        return False
    if payload.get("combo_indices") != combo_indices:
        return False
    records = payload.get("records", [])
    return len(records) == len(combo_indices)


def _evaluate_combo(job):
    combo_index, probe_indices, metric = job
    t0 = time.time()
    predictions, metrics, score = evaluate_probe_set(
        probe_indices,
        predict_benchpress_scores,
        metric=metric,
    )
    probe_ids = [BENCH_IDS[j] for j in probe_indices]
    return {
        "combo_index": int(combo_index),
        "probe_set": probe_ids,
        "probe_names": [BENCH_NAMES.get(bid, bid) for bid in probe_ids],
        "score": float(score) if np.isfinite(score) else None,
        "medape": float(metrics["medape"]) if np.isfinite(metrics["medape"]) else None,
        "medae": float(metrics["medae"]) if np.isfinite(metrics["medae"]) else None,
        "n": int(metrics["n"]),
        "elapsed_s": time.time() - t0,
        "predictions": pack_probe_predictions(predictions),
    }


def _write_config(out_dir: str, config: dict, overwrite: bool = False) -> None:
    run_config = dict(config)
    for key in ("wave_index", "shard_index", "assignment_residue", "assigned_combinations"):
        run_config.pop(key, None)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.json")
    if os.path.exists(path) and not overwrite:
        existing = load_json(path)
        if existing != run_config:
            raise RuntimeError(
                f"Existing {path} has incompatible config. Use a different --out-dir "
                "or delete the stale run directory."
            )
        return
    write_json_atomic(path, run_config, indent=2)


def iter_assigned_combos(
    fixed_indices: list[int],
    remaining_candidates: list[int],
    choose_size: int,
    residue: int,
    modulus: int,
):
    for combo_index, combo in enumerate(itertools.combinations(remaining_candidates, choose_size)):
        if combo_index % modulus != residue:
            continue
        yield combo_index, tuple(fixed_indices) + tuple(combo)


def run_shard(args) -> None:
    candidates, candidate_ids = _load_candidates(args.candidate_allowlist, args.candidate_limit)
    candidate_label = _candidate_source_label(args.candidate_allowlist)
    out_dir = os.path.abspath(args.out_dir or _default_out_dir(args, candidate_label))
    config, fixed_indices, remaining_candidates = _build_config(args, candidates, candidate_ids)
    _write_config(out_dir, config)

    print("=== Exhaustive probe shard ===", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(
        f"k={config['k']} fixed={config['fixed_probe_ids']} "
        f"choose_size={config['choose_size_after_fixed']}",
        flush=True,
    )
    print(
        f"total={config['total_combinations']} assigned={config['assigned_combinations']} "
        f"wave={config['wave_index']}/{config['num_waves']} "
        f"shard={config['shard_index']}/{config['num_shards']} workers={args.workers}",
        flush=True,
    )

    chunk_size = int(args.chunk_size)
    max_subsets = args.max_subsets
    assigned_seen = 0
    evaluated = 0
    skipped_chunks = 0
    t_all = time.time()

    chunk_jobs = []

    def flush_chunk(jobs):
        nonlocal evaluated, skipped_chunks
        if not jobs:
            return
        chunk_index = jobs[0][0]
        combo_indices = [job[1] for job in jobs]
        path = _chunk_path(out_dir, config["wave_index"], config["shard_index"], chunk_index)
        if _is_valid_existing_chunk(path, config, combo_indices):
            skipped_chunks += 1
            print(f"skip chunk {chunk_index:06d} ({len(jobs)} combos)", flush=True)
            return

        print(f"run chunk {chunk_index:06d} ({len(jobs)} combos) -> {path}", flush=True)
        t0 = time.time()
        records = []
        worker_jobs = [(combo_index, probe_indices, args.metric) for _, combo_index, probe_indices in jobs]
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(SEED,),
        ) as pool:
            futures = [pool.submit(_evaluate_combo, job) for job in worker_jobs]
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda row: row["combo_index"])
        payload = {
            "config": _config_for_chunk(config),
            "combo_indices": combo_indices,
            "records": records,
            "elapsed_s": time.time() - t0,
        }
        write_json_atomic(path, payload)
        evaluated += len(records)
        print(f"wrote chunk {chunk_index:06d}: {len(records)} records", flush=True)

    choose_size = config["choose_size_after_fixed"]
    residue = config["assignment_residue"]
    modulus = config["assignment_modulus"]
    for combo_index, probe_indices in iter_assigned_combos(
        fixed_indices, remaining_candidates, choose_size, residue, modulus
    ):
        if max_subsets is not None and assigned_seen >= int(max_subsets):
            break
        chunk_index = assigned_seen // chunk_size
        chunk_jobs.append((chunk_index, combo_index, probe_indices))
        assigned_seen += 1
        if len(chunk_jobs) >= chunk_size:
            flush_chunk(chunk_jobs)
            chunk_jobs = []
    flush_chunk(chunk_jobs)

    print(
        f"done: assigned_seen={assigned_seen} evaluated={evaluated} "
        f"skipped_chunks={skipped_chunks} elapsed_s={time.time() - t_all:.1f}",
        flush=True,
    )


def _load_run_config(out_dir: str) -> dict:
    path = os.path.join(out_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config: {path}")
    return load_json(path)


def _expected_chunk_count(assigned_count: int, chunk_size: int) -> int:
    return int(math.ceil(assigned_count / chunk_size)) if assigned_count else 0


def merge(args) -> None:
    out_dir = os.path.abspath(args.out_dir)
    base_config = _load_run_config(out_dir)
    num_waves = int(base_config["num_waves"])
    num_shards = int(base_config["num_shards"])
    total = int(base_config["total_combinations"])
    chunk_size = int(base_config["chunk_size"])
    modulus = num_waves * num_shards

    summaries = []
    missing = []
    seen = set()
    best = None
    for wave_index in range(num_waves):
        for shard_index in range(num_shards):
            residue = _assignment_residue(wave_index, num_waves, shard_index, num_shards)
            assigned = _assigned_count(total, residue, modulus)
            n_chunks = _expected_chunk_count(assigned, chunk_size)
            for chunk_index in range(n_chunks):
                path = _chunk_path(out_dir, wave_index, shard_index, chunk_index)
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                payload = load_json(path)
                for record in payload.get("records", []):
                    combo_index = int(record["combo_index"])
                    if combo_index in seen:
                        raise RuntimeError(f"Duplicate combo_index {combo_index} in {path}")
                    seen.add(combo_index)
                    summary = {
                        "combo_index": combo_index,
                        "probe_set": record["probe_set"],
                        "score": record["score"],
                        "medape": record["medape"],
                        "medae": record["medae"],
                        "n": record["n"],
                        "elapsed_s": record.get("elapsed_s"),
                    }
                    summaries.append(summary)
                    if record["score"] is not None and (
                        best is None or record["score"] < best["score"]
                    ):
                        best = dict(summary)

    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Missing {len(missing)} chunks. First missing: {missing[0]}. "
            "Use --allow-incomplete only for diagnostics."
        )
    expected = total if not args.allow_incomplete else len(seen)
    if len(seen) != expected and not args.allow_incomplete:
        raise RuntimeError(f"Expected {expected} combo records, found {len(seen)}")

    summaries.sort(key=lambda row: (float("inf") if row["score"] is None else row["score"], row["combo_index"]))
    top_n = int(args.top_n)
    payload = {
        "config": base_config,
        "complete": not missing and len(seen) == total,
        "n_records": len(seen),
        "missing_chunks": missing,
        "best": best,
        "top": summaries[:top_n],
    }
    out_path = os.path.join(out_dir, "merged_summary.json.gz")
    write_json_atomic(out_path, payload, indent=2)
    print(f"merged -> {out_path}", flush=True)
    if best:
        print(f"best score={best['score']} probe_set={best['probe_set']}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-shard")
    run.add_argument("--candidate-allowlist", type=str, default=None)
    run.add_argument("--candidate-limit", type=int, default=None)
    run.add_argument("--fixed-probe", type=str, default=None, help="comma-separated benchmark IDs")
    run.add_argument("--k", type=int, default=5)
    run.add_argument("--metric", type=str, default="medae", choices=["medae", "medape"])
    run.add_argument("--num-waves", type=int, default=10)
    run.add_argument("--wave-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    run.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    run.add_argument("--max-subsets", type=int, default=None)
    run.add_argument("--out-dir", type=str, default=None)
    run.set_defaults(func=run_shard)

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--out-dir", type=str, required=True)
    merge_parser.add_argument("--top-n", type=int, default=100)
    merge_parser.add_argument("--allow-incomplete", action="store_true")
    merge_parser.set_defaults(func=merge)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
