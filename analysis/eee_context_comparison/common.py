"""Shared matrix loading and mapping for the EEE context comparison."""

import hashlib
import os

import numpy as np

from benchpress.data.score_matrix import ScoreMatrix
from benchpress.evaluation_harness import (
    BENCH_IDS,
    MODEL_IDS,
    M_FULL,
)


MIN_BENCHMARKS_PER_MODEL = 15
MIN_MODELS_PER_BENCHMARK = 8
MIN_MATCHED_TARGETS = 30
PCT_TYPES = {"pct", "percent", "percentage"}


def file_sha256(path):
    """Return the SHA-256 digest of one input file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_benchpress_json_path():
    """Return the JSON path used by the installed BenchPress package."""
    configured = os.environ.get("BENCHPRESS_DATA")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    import benchpress
    return os.path.join(
        os.path.dirname(benchpress.__file__),
        "data",
        "llm_benchmark_data.json",
    )


def load_filtered_eee(eee_dir):
    """Load EEE percentage columns and apply the established observation filter."""
    scores_path = os.path.join(os.path.expanduser(eee_dir), "scores.csv")
    score_matrix = ScoreMatrix.from_file(scores_path)
    values = np.asarray(score_matrix.values, dtype=float)
    model_ids = list(score_matrix.model_ids)
    benchmark_ids = list(score_matrix.benchmark_ids)
    metric = dict(score_matrix.metric)

    pct = np.asarray([
        str(metric[benchmark_id].get("type", "")).lower() in PCT_TYPES
        for benchmark_id in benchmark_ids
    ])
    values = values[:, pct]
    benchmark_ids = [
        benchmark_id
        for benchmark_id, keep in zip(benchmark_ids, pct)
        if keep
    ]
    metric = {benchmark_id: metric[benchmark_id] for benchmark_id in benchmark_ids}

    while True:
        observed = np.isfinite(values)
        row_ok = observed.sum(axis=1) >= MIN_BENCHMARKS_PER_MODEL
        col_ok = observed.sum(axis=0) >= MIN_MODELS_PER_BENCHMARK
        if row_ok.all() and col_ok.all():
            break
        values = values[row_ok][:, col_ok]
        model_ids = [
            model_id for model_id, keep in zip(model_ids, row_ok) if keep
        ]
        benchmark_ids = [
            benchmark_id
            for benchmark_id, keep in zip(benchmark_ids, col_ok)
            if keep
        ]
        metric = {
            benchmark_id: metric[benchmark_id]
            for benchmark_id in benchmark_ids
        }
        if values.size == 0:
            raise RuntimeError("EEE observation filtering removed the entire matrix")

    return {
        "values": values,
        "model_ids": model_ids,
        "benchmark_ids": benchmark_ids,
        "metric": metric,
        "scores_path": scores_path,
    }


def mapping_pairs(mapping, key):
    """Validate and return unique explicit mapping pairs for one axis."""
    rows = mapping.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"mapping must contain a non-empty {key!r} list")
    pairs = []
    source_seen = set()
    target_seen = set()
    for row in rows:
        source_id = row.get("benchpress_id")
        target_id = row.get("eee_id")
        if not source_id or not target_id:
            raise ValueError(f"each {key} mapping needs benchpress_id and eee_id")
        if source_id in source_seen:
            raise ValueError(f"duplicate BenchPress {key} mapping: {source_id}")
        if target_id in target_seen:
            raise ValueError(f"duplicate EEE {key} mapping: {target_id}")
        source_seen.add(source_id)
        target_seen.add(target_id)
        pairs.append((source_id, target_id))
    return pairs


def build_matched_targets(eee, mapping):
    """Return mapped target cells observed in both matrices."""
    model_pairs = mapping_pairs(mapping, "models")
    benchmark_pairs = mapping_pairs(mapping, "benchmarks")
    bp_model_index = {value: idx for idx, value in enumerate(MODEL_IDS)}
    bp_benchmark_index = {value: idx for idx, value in enumerate(BENCH_IDS)}
    eee_model_index = {
        value: idx for idx, value in enumerate(eee["model_ids"])
    }
    eee_benchmark_index = {
        value: idx for idx, value in enumerate(eee["benchmark_ids"])
    }

    for bp_id, eee_id in model_pairs:
        if bp_id not in bp_model_index:
            raise ValueError(f"unknown BenchPress model mapping: {bp_id}")
        if eee_id not in eee_model_index:
            raise ValueError(f"EEE model did not survive filtering: {eee_id}")
    for bp_id, eee_id in benchmark_pairs:
        if bp_id not in bp_benchmark_index:
            raise ValueError(f"unknown BenchPress benchmark mapping: {bp_id}")
        if eee_id not in eee_benchmark_index:
            raise ValueError(f"EEE benchmark did not survive filtering: {eee_id}")

    target_matrix = np.full(
        (len(model_pairs), len(benchmark_pairs)), np.nan, dtype=float)
    target_metadata = {}
    for mapped_i, (bp_model_id, eee_model_id) in enumerate(model_pairs):
        for mapped_j, (bp_benchmark_id, eee_benchmark_id) in enumerate(
                benchmark_pairs):
            bp_i = bp_model_index[bp_model_id]
            bp_j = bp_benchmark_index[bp_benchmark_id]
            eee_i = eee_model_index[eee_model_id]
            eee_j = eee_benchmark_index[eee_benchmark_id]
            bp_score = float(M_FULL[bp_i, bp_j])
            eee_score = float(eee["values"][eee_i, eee_j])
            if not np.isfinite(bp_score) or not np.isfinite(eee_score):
                continue
            target_matrix[mapped_i, mapped_j] = bp_score
            target_metadata[(mapped_i, mapped_j)] = {
                "target_key": f"{bp_model_id}/{bp_benchmark_id}",
                "benchpress_model_id": bp_model_id,
                "benchpress_benchmark_id": bp_benchmark_id,
                "eee_model_id": eee_model_id,
                "eee_benchmark_id": eee_benchmark_id,
                "benchpress_cell": [int(bp_i), int(bp_j)],
                "eee_cell": [int(eee_i), int(eee_j)],
                "actual": bp_score,
                "eee_observed_score": eee_score,
            }

    if len(target_metadata) < MIN_MATCHED_TARGETS:
        raise RuntimeError(
            f"only {len(target_metadata)} matched observed target cells; "
            f"need at least {MIN_MATCHED_TARGETS}")
    return target_matrix, target_metadata
