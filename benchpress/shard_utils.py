#!/usr/bin/env python3
"""Shared result and shard helpers for BenchPress experiment scripts.

Includes per-(k, seed) shard naming/writing/merging plus reusable helpers
for turning hyperparameter dicts into stable, filesystem-safe shard ids.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable, Iterable

import numpy as np

from benchpress.io_utils import load_json, write_json, write_json_atomic


# ── Hyperparameter / shard-id helpers ────────────────────────────────

def slug(value):
    """Lowercase, filesystem-safe slug for human-readable strings.

    Replaces spaces and hyphens with underscores; suitable for embedding
    method/transform names in shard ids. For arbitrary user strings prefer
    `benchpress.io_utils.safe_token`.
    """
    return str(value).lower().replace(" ", "_").replace("-", "_")


def canonical_hp_json(hp):
    """Stable, compact JSON serialization for a hyperparameter dict."""
    return json.dumps(hp, sort_keys=True, separators=(",", ":"))


def short_text_hash(text, n=10, algorithm="sha1"):
    """Short stable text hash for cache/shard identifiers."""
    if algorithm == "sha1":
        digest = hashlib.sha1(str(text).encode("utf-8"))
    elif algorithm == "md5":
        digest = hashlib.md5(str(text).encode("utf-8"))
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    return digest.hexdigest()[:n]


def hp_short_hash(hp, n=10):
    """Short SHA1 hash of a hyperparameter dict for shard ids."""
    return short_text_hash(canonical_hp_json(hp), n=n)


def prediction_error_summary_by_k_seed(raw_predictions):
    """Summarize raw prediction rows by `(k, seed)` using canonical error metrics."""
    from benchpress.evaluation_harness import compute_prediction_error

    grouped = {}
    for row in raw_predictions:
        key = (int(row["k"]), int(row["seed"]))
        grouped.setdefault(key, {"actual": [], "pred": []})
        grouped[key]["actual"].append(float(row["actual"]))
        grouped[key]["pred"].append(float(row["pred"]))

    rows = []
    for (k, seed_idx), vals in sorted(grouped.items()):
        metrics = compute_prediction_error(
            np.array(vals["actual"], dtype=float),
            np.array(vals["pred"], dtype=float),
            aggregation="pool",
        )
        rows.append({
            "k": k,
            "seed": seed_idx,
            "n": int(metrics["n"]),
            "medape": float(metrics["medape"]),
            "medae": float(metrics["medae"]),
        })
    return rows


def write_prediction_result(
    raw_predictions,
    output_path,
    config,
    include_summary_by_k_seed=False,
    indent=None,
):
    """Write the common BenchPress prediction-result payload shape."""
    output = {"config": config}
    if include_summary_by_k_seed:
        output["summary_by_k_seed"] = prediction_error_summary_by_k_seed(raw_predictions)
    output["raw_predictions"] = raw_predictions
    write_json(output_path, output, indent=indent)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved -> {output_path} ({size_mb:.1f} MB)", flush=True)
    return output


def default_k_seed_shard_name(k, seed_idx, suffix="", extension=".json"):
    """Default shard filename for one `(k, seed)` unit."""
    return f"k{k}_s{seed_idx}{suffix}{extension}"


def parse_k_seed_shard_name(name):
    """Parse `k{K}_s{SEED}[...].json[.gz]` shard filenames."""
    stem = name.removesuffix(".json.gz").removesuffix(".json")
    parts = stem.split("_")
    if len(parts) < 2 or not parts[0].startswith("k") or not parts[1].startswith("s"):
        raise ValueError(f"Cannot parse k/seed from shard filename: {name}")
    return int(parts[0][1:]), int(parts[1][1:])


def expected_k_seed_shard_names(k_values, n_seeds, shard_name_fn):
    return {
        shard_name_fn(k, seed_idx)
        for k in k_values
        for seed_idx in range(n_seeds)
    }


def assert_expected_shard_names(paths, expected_names, shard_dir, hint=None):
    actual_names = {os.path.basename(path) for path in paths}
    extra = sorted(actual_names - expected_names)
    missing = sorted(expected_names - actual_names)
    if extra or missing:
        msg = (
            f"Shard set mismatch in {shard_dir}: "
            f"missing={missing[:10]} extra={extra[:10]}."
        )
        if hint:
            msg = f"{msg} {hint}"
        raise RuntimeError(msg)


def validate_k_seed_result_payload(
    data,
    path,
    expected_k,
    expected_seed,
    expected_config=None,
    expected_summary_n=None,
    require_summary=False,
):
    """Validate a one-shard prediction-result payload before reuse or merge."""
    if expected_config is not None:
        config = data.get("config", {})
        actual_config = {key: config.get(key) for key in expected_config}
        if actual_config != expected_config:
            raise ValueError(
                f"{path} config mismatch: expected {expected_config}, found {actual_config}"
            )

    rows = data.get("summary_by_k_seed", [])
    if require_summary:
        if len(rows) != 1:
            raise ValueError(
                f"{path} must contain exactly one summary row, found {len(rows)}"
            )
        row = rows[0]
        if int(row.get("k", -1)) != expected_k or int(row.get("seed", -1)) != expected_seed:
            raise ValueError(
                f"{path} summary key mismatch: expected k={expected_k}, "
                f"seed={expected_seed}, found k={row.get('k')}, seed={row.get('seed')}"
            )
        if expected_summary_n is not None and int(row.get("n", -1)) != expected_summary_n:
            raise ValueError(
                f"{path} denominator mismatch: expected n={expected_summary_n}, "
                f"found n={row.get('n')}"
            )

    for raw in data.get("raw_predictions", []):
        if int(raw.get("k", -1)) != expected_k or int(raw.get("seed", -1)) != expected_seed:
            raise ValueError(
                f"{path} raw row mismatch: expected k={expected_k}, seed={expected_seed}, "
                f"found k={raw.get('k')}, seed={raw.get('seed')}"
            )


@dataclass(frozen=True)
class KSeedShardJob:
    k: int
    seed_idx: int
    output_path: str
    write_shard_fn: Callable
    validate_shard_fn: Callable | None = None
    kwargs: dict | None = None


def _run_k_seed_shard_job(job):
    kwargs = job.kwargs or {}
    if os.path.exists(job.output_path) and os.path.getsize(job.output_path) > 100:
        try:
            if job.validate_shard_fn is not None:
                job.validate_shard_fn(
                    load_json(job.output_path),
                    job.output_path,
                    job.k,
                    job.seed_idx,
                    **kwargs,
                )
            return job.k, job.seed_idx, "cached", 0.0
        except Exception as exc:
            print(
                f"  k={job.k:2d} seed={job.seed_idx}: stale cache ignored ({exc})",
                flush=True,
            )

    start = time.time()
    try:
        job.write_shard_fn(job.k, job.seed_idx, job.output_path, **kwargs)
    except Exception as exc:
        return job.k, job.seed_idx, f"FAIL: {exc}", time.time() - start
    return job.k, job.seed_idx, "ok", time.time() - start


def run_k_seed_shards(
    k_values: Iterable[int],
    n_seeds: int,
    shard_dir: str,
    workers: int,
    shard_name_fn: Callable,
    write_shard_fn: Callable,
    validate_shard_fn: Callable | None = None,
    job_kwargs: dict | None = None,
    label_extra: str = "",
):
    """Run or reuse one JSON shard for each `(k, seed)` unit."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    os.makedirs(shard_dir, exist_ok=True)
    k_values = list(k_values)
    jobs = [
        KSeedShardJob(
            k=k,
            seed_idx=seed_idx,
            output_path=os.path.join(shard_dir, shard_name_fn(k, seed_idx)),
            write_shard_fn=write_shard_fn,
            validate_shard_fn=validate_shard_fn,
            kwargs=job_kwargs or {},
        )
        for k in k_values
        for seed_idx in range(n_seeds)
    ]
    if not jobs:
        return

    pool_size = min(workers, len(jobs))
    extra = f", {label_extra}" if label_extra else ""
    print(
        f"k={k_values}, seeds={n_seeds}, workers={pool_size}, shard_dir={shard_dir}{extra}",
        flush=True,
    )
    start = time.time()
    if pool_size == 1:
        iterator = map(_run_k_seed_shard_job, jobs)
    else:
        with Pool(pool_size) as pool:
            iterator = pool.imap_unordered(_run_k_seed_shard_job, jobs)
            for k, seed_idx, status, dt in iterator:
                print(
                    f"  k={k:2d} seed={seed_idx}: {status} "
                    f"({dt:.0f}s, wall {time.time() - start:.0f}s)",
                    flush=True,
                )
            return

    for k, seed_idx, status, dt in iterator:
        print(
            f"  k={k:2d} seed={seed_idx}: {status} "
            f"({dt:.0f}s, wall {time.time() - start:.0f}s)",
            flush=True,
        )


def merge_prediction_shards(
    shard_dir,
    expected_names,
    validate_shard_fn,
    validation_kwargs=None,
    key_fields=("k", "seed", "model", "bench"),
    sort_fields=("k", "seed", "model", "bench"),
    shard_glob="k*_s*.json*",
):
    """Load, validate, de-duplicate, and merge raw prediction shards."""
    paths = sorted(glob.glob(os.path.join(shard_dir, shard_glob)))
    if not paths:
        raise FileNotFoundError(f"No shards found in {shard_dir}")
    assert_expected_shard_names(paths, expected_names, shard_dir)

    raw = []
    seen = set()
    validation_kwargs = validation_kwargs or {}
    for path in paths:
        expected_k, expected_seed = parse_k_seed_shard_name(os.path.basename(path))
        data = load_json(path)
        validate_shard_fn(data, path, expected_k, expected_seed, **validation_kwargs)
        for row in data["raw_predictions"]:
            key = tuple(row[field] for field in key_fields)
            if key in seen:
                raise RuntimeError(f"Duplicate raw prediction key while merging: {key}")
            seen.add(key)
            raw.append(row)

    raw.sort(key=lambda row: tuple(row[field] for field in sort_fields))
    print(f"Merged {len(paths)} shards -> {len(raw)} raw rows", flush=True)
    return raw


def load_json_unit_shards(
    shards_dir,
    suffix=".json",
    sort_key=lambda unit: int(unit["unit_index"]),
    label="shard JSON files",
):
    """Load a directory of JSON unit shards and sort them deterministically."""
    if not os.path.isdir(shards_dir):
        raise FileNotFoundError(f"No {label} found in {shards_dir}")
    paths = [
        os.path.join(shards_dir, name)
        for name in os.listdir(shards_dir)
        if name.endswith(suffix)
    ]
    units = [load_json(path) for path in paths]
    if sort_key is not None:
        units.sort(key=sort_key)
    if not units:
        raise FileNotFoundError(f"No {label} found in {shards_dir}")
    return units


def unit_shard_path(shards_dir, prefix, unit_id, seed):
    """Path for a per-unit JSON shard keyed by a model/benchmark id and seed."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(unit_id))
    return os.path.join(shards_dir, f"{prefix}_{safe_id}_seed_{seed}.json")


def write_unit_json_atomic(shards_dir, prefix, unit_id, seed, payload):
    """Atomically write one per-unit shard and return its path."""
    path = unit_shard_path(shards_dir, prefix, unit_id, seed)
    write_json_atomic(path, payload)
    return path


def collect_unit_records(units, records_key="records"):
    """Flatten record lists from loaded unit shards."""
    records = []
    for unit in units:
        records.extend(unit[records_key])
    return records
