"""Generate score-independent EEE mapping candidates."""

import difflib
import os
import re

import numpy as np

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_NAMES,
    MODEL_IDS,
    MODEL_NAMES,
    M_FULL,
)
from benchpress.io_utils import write_json_atomic

from common import MIN_BENCHMARKS_PER_MODEL, MIN_MODELS_PER_BENCHMARK


def identifier_tokens(value):
    """Return normalized alphanumeric terms for candidate matching."""
    text = str(value).lower()
    text = text.replace("τ", "tau")
    text = text.replace("²", "2").replace("³", "3")
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)


def candidate_similarity(source_id, source_name, candidate_id):
    """Score one lexical mapping candidate without using score values."""
    source_tokens = identifier_tokens(f"{source_id} {source_name}")
    candidate_tokens = identifier_tokens(candidate_id)
    source_set = set(source_tokens)
    candidate_set = set(candidate_tokens)
    shared = source_set & candidate_set
    if not shared:
        return 0.0

    source_compact = "".join(identifier_tokens(source_id))
    name_compact = "".join(identifier_tokens(source_name))
    candidate_compact = "".join(candidate_tokens)
    sequence = max(
        difflib.SequenceMatcher(None, source_compact, candidate_compact).ratio(),
        difflib.SequenceMatcher(None, name_compact, candidate_compact).ratio(),
    )
    jaccard = len(shared) / len(source_set | candidate_set)
    containment = 0.0
    if source_compact and source_compact in candidate_compact:
        containment = 1.0
    elif name_compact and name_compact in candidate_compact:
        containment = 0.9
    return 0.5 * sequence + 0.3 * jaccard + 0.2 * containment


def top_candidates(source_ids, source_names, candidate_ids, observed_counts,
                   limit=8):
    """Return top lexical candidates and coverage for each source identifier."""
    report = {}
    for source_id in source_ids:
        ranked = []
        for candidate_id, observed_count in zip(candidate_ids, observed_counts):
            similarity = candidate_similarity(
                source_id, source_names[source_id], candidate_id)
            if similarity <= 0:
                continue
            ranked.append({
                "id": candidate_id,
                "similarity": float(similarity),
                "observed_cells": int(observed_count),
            })
        ranked.sort(
            key=lambda row: (row["similarity"], row["observed_cells"]),
            reverse=True,
        )
        report[source_id] = {
            "name": source_names[source_id],
            "candidates": ranked[:limit],
        }
    return report


def write_candidate_report(eee, output):
    """Write mapping candidates from matrix identifiers and coverage only."""
    os.makedirs(output, exist_ok=True)
    values = eee["values"]
    report = {
        "protocol": {
            "candidate_selection": "lexical similarity plus observed-cell coverage",
            "score_values_used": False,
            "eee_filter": {
                "percentage_columns_only": True,
                "min_benchmarks_per_model": MIN_BENCHMARKS_PER_MODEL,
                "min_models_per_benchmark": MIN_MODELS_PER_BENCHMARK,
            },
        },
        "matrices": {
            "benchpress": {
                "shape": [int(M_FULL.shape[0]), int(M_FULL.shape[1])],
                "observed_cells": int(np.isfinite(M_FULL).sum()),
            },
            "eee": {
                "shape": [int(values.shape[0]), int(values.shape[1])],
                "observed_cells": int(np.isfinite(values).sum()),
            },
        },
        "models": top_candidates(
            MODEL_IDS,
            MODEL_NAMES,
            eee["model_ids"],
            np.isfinite(values).sum(axis=1),
        ),
        "benchmarks": top_candidates(
            BENCH_IDS,
            BENCH_NAMES,
            eee["benchmark_ids"],
            np.isfinite(values).sum(axis=0),
        ),
    }
    path = os.path.join(output, "candidate_report.json")
    write_json_atomic(path, report, indent=2, sort_keys=True, trailing_newline=True)
    print(f"wrote {path}", flush=True)
