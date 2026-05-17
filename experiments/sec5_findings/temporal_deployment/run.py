#!/usr/bin/env python
"""Temporal deployment stress test for newly released model families.

For each landmark family, train on only models released before the family
cutoff. Reveal k observed benchmark scores per target model, predict the
remaining observed target scores, and write every observed target cell.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.build_benchmark_matrix import MODELS
from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_NAMES,
    M_FULL,
    MODEL_IDS,
    MODEL_IDX,
    MODEL_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    compute_prediction_error,
)
from benchpress.io_utils import load_json, safe_token, write_json, write_json_atomic
from benchpress.methods.predictors import predict_benchpress_scores
from benchpress.shard_utils import short_text_hash

PROTOCOL_VERSION = "temporal_deployment_all_observed_v3"
BASE_SEED = 42
K_VALUES = [1, 3, 5, 8, 10, 15]
N_SEEDS = 10

RESULTS_DIR = os.path.join(HERE, "results")
SHARD_DIR = os.path.join(RESULTS_DIR, "shards")
RESULTS_PATH = os.path.join(HERE, "results.json")


LANDMARKS = [
    {
        "family_key": "deepseek_r1",
        "family_name": "DeepSeek R1",
        "cutoff_date": "2025-01-20",
        "match": {"startswith": ["deepseek-r1"], "exclude_contains": ["0528"]},
    },
    {
        "family_key": "gemini_2_5_pro",
        "family_name": "Gemini 2.5 Pro",
        "cutoff_date": "2025-03-25",
        "match": {"ids": ["gemini-2.5-pro"]},
    },
    {
        "family_key": "gpt_4_1_family",
        "family_name": "GPT-4.1 family",
        "cutoff_date": "2025-04-14",
        "match": {"startswith": ["gpt-4.1"]},
    },
    {
        "family_key": "qwen_3",
        "family_name": "Qwen 3",
        "cutoff_date": "2025-05-15",
        "match": {"startswith": ["qwen3-"]},
    },
    {
        "family_key": "claude_sonnet_opus_4",
        "family_name": "Claude Sonnet/Opus 4",
        "cutoff_date": "2025-05-22",
        "match": {"ids": ["claude-sonnet-4", "claude-opus-4"]},
    },
    {
        "family_key": "gpt_5",
        "family_name": "GPT-5",
        "cutoff_date": "2025-08-01",
        "match": {"ids": ["gpt-5"]},
    },
    {
        "family_key": "claude_sonnet_4_5",
        "family_name": "Claude Sonnet 4.5",
        "cutoff_date": "2025-09-29",
        "match": {"ids": ["claude-sonnet-4.5"]},
    },
    {
        "family_key": "gpt_5_1",
        "family_name": "GPT-5.1",
        "cutoff_date": "2025-11-13",
        "match": {"ids": ["gpt-5.1"]},
    },
]


MODEL_RELEASE_DATES = {
    m[0]: m[3]
    for m in MODELS
    if len(m) > 3 and m[3] and m[0] in MODEL_IDX
}


def model_matches(model_id: str, rule: dict) -> bool:
    ids = set(rule.get("ids", []))
    if ids and model_id not in ids:
        return False
    prefixes = rule.get("startswith", [])
    if prefixes and not any(model_id.startswith(prefix) for prefix in prefixes):
        return False
    excluded = rule.get("exclude_contains", [])
    return not any(token in model_id for token in excluded)


def landmark_by_key(family_key: str) -> dict:
    for landmark in LANDMARKS:
        if landmark["family_key"] == family_key:
            return landmark
    raise ValueError(f"Unknown family key {family_key!r}")


def target_model_ids(landmark: dict) -> list[str]:
    return [
        mid
        for mid in MODEL_IDS
        if model_matches(mid, landmark["match"]) and mid in MODEL_RELEASE_DATES
    ]


def train_model_ids(landmark: dict, target_ids: list[str]) -> list[str]:
    target_set = set(target_ids)
    cutoff = landmark["cutoff_date"]
    return [
        mid
        for mid, release_date in sorted(MODEL_RELEASE_DATES.items(), key=lambda x: x[1])
        if release_date < cutoff and mid not in target_set
    ]


def target_candidate_benchmarks(target_ids: list[str]) -> list[int]:
    candidates = set()
    for mid in target_ids:
        i = MODEL_IDX[mid]
        candidates.update(int(j) for j in np.where(OBSERVED[i])[0])
    return sorted(candidates)


def sample_revealed_by_model(target_ids: list[str], k: int, rng: np.random.RandomState) -> dict[str, list[int]]:
    revealed_by_model = {}
    for mid in target_ids:
        i = MODEL_IDX[mid]
        obs_j = np.where(OBSERVED[i])[0].astype(int)
        if len(obs_j) < int(k):
            continue
        shuffled = obs_j.copy()
        rng.shuffle(shuffled)
        revealed_by_model[mid] = [int(j) for j in shuffled[: int(k)]]
    return revealed_by_model


def shard_path(family_key: str, k: int, seed: int) -> str:
    return os.path.join(
        SHARD_DIR,
        f"{safe_token(family_key)}__k{int(k)}__s{int(seed)}.json.gz",
    )


def shard_config(landmark: dict, k: int, seed: int) -> dict:
    target_ids = target_model_ids(landmark)
    train_ids = train_model_ids(landmark, target_ids)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "family_key": landmark["family_key"],
        "family_name": landmark["family_name"],
        "cutoff_date": landmark["cutoff_date"],
        "k": int(k),
        "seed": int(seed),
        "base_seed": BASE_SEED,
        "predictor": "predict_benchpress_scores",
        "train_rule": "models with release_date strictly before cutoff_date",
        "probe_rule": "for each target model, randomly reveal k of that model's observed benchmark scores",
        "metric_rule": "pool observed target cells with finite predictions; revealed cells are exact zero-error predictions; non-predictable cells are recorded separately",
        "matrix_shape": [int(N_MODELS), int(N_BENCH)],
        "target_model_ids": target_ids,
        "train_model_ids": train_ids,
    }


def _finite_float(value):
    return float(value) if np.isfinite(value) else None


def validate_shard(payload: dict, path: str, family_key: str, k: int, seed: int):
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "family_key": family_key,
        "k": int(k),
        "seed": int(seed),
        "matrix_shape": [int(N_MODELS), int(N_BENCH)],
    }
    config = payload.get("config", {})
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"{path} config mismatch: expected {expected}, found {actual}")
    for row in payload.get("raw_predictions", []):
        if row.get("family_key") != family_key or int(row.get("k", -1)) != int(k) or int(row.get("seed", -1)) != int(seed):
            raise ValueError(f"{path} contains a raw row for the wrong unit: {row}")


def run_shard(family_key: str, k: int, seed: int, out_path: Optional[str] = None) -> dict:
    landmark = landmark_by_key(family_key)
    config = shard_config(landmark, k, seed)
    target_ids = config["target_model_ids"]
    train_ids = config["train_model_ids"]
    if not target_ids:
        raise RuntimeError(f"{family_key}: no target models matched")
    if not train_ids:
        raise RuntimeError(f"{family_key}: no training models before cutoff")

    rng_seed_material = f"{BASE_SEED}:{family_key}:{int(seed)}"
    rng_seed = int(short_text_hash(rng_seed_material, n=8), 16)
    rng = np.random.RandomState(rng_seed)

    revealed_by_model = sample_revealed_by_model(target_ids, k, rng)
    if not revealed_by_model:
        raise RuntimeError(
            f"{family_key}: no target model has at least k observed scores for k={k}"
        )

    M_train = np.full_like(M_FULL, np.nan, dtype=float)
    for mid in train_ids:
        i = MODEL_IDX[mid]
        M_train[i, :] = M_FULL[i, :]
    for mid, revealed_js in revealed_by_model.items():
        i = MODEL_IDX[mid]
        for j in revealed_js:
            M_train[i, j] = M_FULL[i, j]

    eval_cells = []
    for mid, revealed_js in revealed_by_model.items():
        i = MODEL_IDX[mid]
        revealed_set = set(revealed_js)
        for j in np.where(OBSERVED[i])[0]:
            j = int(j)
            eval_cells.append((i, j, j in revealed_set))
    if len(eval_cells) < 3:
        raise RuntimeError(f"{family_key} k={k} seed={seed}: fewer than 3 observed cells")

    M_pred = predict_benchpress_scores(M_train)
    metric_cells = [
        (i, j, is_revealed)
        for i, j, is_revealed in eval_cells
        if is_revealed or np.isfinite(M_pred[i, j])
    ]
    if len(metric_cells) < 3:
        raise RuntimeError(f"{family_key} k={k} seed={seed}: fewer than 3 metric cells")

    actual = np.array([M_FULL[i, j] for i, j, _ in metric_cells], dtype=float)
    predicted = np.array([
        M_FULL[i, j] if is_revealed else M_pred[i, j]
        for i, j, is_revealed in metric_cells
    ], dtype=float)
    metrics = compute_prediction_error(actual, predicted, aggregation="pool")
    if int(metrics["n"]) != len(metric_cells):
        raise RuntimeError(
            f"{family_key} k={k} seed={seed}: metric denominator mismatch "
            f"({metrics['n']} != {len(metric_cells)})"
        )

    raw_predictions = []
    for i, j, is_revealed in eval_cells:
        pred_value = float(M_FULL[i, j]) if is_revealed else float(M_pred[i, j])
        pred = pred_value if np.isfinite(pred_value) else None
        raw_predictions.append({
            "family_key": family_key,
            "family_name": landmark["family_name"],
            "cutoff_date": landmark["cutoff_date"],
            "k": int(k),
            "seed": int(seed),
            "model_id": MODEL_IDS[i],
            "model_name": MODEL_NAMES[MODEL_IDS[i]],
            "benchmark_id": BENCH_IDS[j],
            "benchmark_name": BENCH_NAMES[BENCH_IDS[j]],
            "actual": float(M_FULL[i, j]),
            "pred": pred,
            "is_revealed": bool(is_revealed),
            "is_metric_cell": bool(is_revealed or pred is not None),
            "prediction_source": (
                "revealed" if is_revealed
                else "benchpress" if pred is not None
                else "not_predictable"
            ),
        })
    n_revealed_cells = sum(1 for _, _, is_revealed in eval_cells if is_revealed)
    n_hidden_cells = len(eval_cells) - n_revealed_cells
    n_not_predictable_cells = sum(
        1
        for i, j, is_revealed in eval_cells
        if not is_revealed and not np.isfinite(M_pred[i, j])
    )

    payload = {
        "config": {
            **config,
            "active_target_model_ids": list(revealed_by_model.keys()),
            "revealed_by_model": {
                mid: [BENCH_IDS[j] for j in js]
                for mid, js in revealed_by_model.items()
            },
            "n_eval_cells": int(len(eval_cells)),
            "n_metric_cells": int(len(metric_cells)),
            "n_revealed_cells": int(n_revealed_cells),
            "n_hidden_cells": int(n_hidden_cells),
            "n_not_predictable_cells": int(n_not_predictable_cells),
            "n_raw_predictions": int(len(raw_predictions)),
        },
        "metrics": {
            "n": int(metrics["n"]),
            "medape": _finite_float(metrics["medape"]),
            "medae": _finite_float(metrics["medae"]),
        },
        "raw_predictions": raw_predictions,
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=None)
    return payload


def expected_units(family_keys: list[str], k_values: list[int], n_seeds: int):
    return [
        (family_key, int(k), int(seed))
        for family_key in family_keys
        for k in k_values
        for seed in range(int(n_seeds))
    ]


def _run_or_reuse_unit(args):
    family_key, k, seed = args
    path = shard_path(family_key, k, seed)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        payload = load_json(path)
        try:
            validate_shard(payload, path, family_key, k, seed)
            return family_key, k, seed, "cached"
        except ValueError:
            os.remove(path)
    run_shard(family_key, k, seed, out_path=path)
    return family_key, k, seed, "ok"


def run_all(family_keys: list[str], k_values: list[int], n_seeds: int, workers: int):
    os.makedirs(SHARD_DIR, exist_ok=True)
    units = expected_units(family_keys, k_values, n_seeds)
    print(
        f"Temporal deployment: {len(family_keys)} families, "
        f"k={k_values}, seeds={n_seeds}, workers={workers}",
        flush=True,
    )
    if workers <= 1:
        for unit in units:
            family_key, k, seed, status = _run_or_reuse_unit(unit)
            print(f"  {family_key:24s} k={k:2d} seed={seed}: {status}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_or_reuse_unit, unit) for unit in units]
        for fut in as_completed(futures):
            family_key, k, seed, status = fut.result()
            print(f"  {family_key:24s} k={k:2d} seed={seed}: {status}", flush=True)


def describe_design(family_keys: list[str], k_values: list[int], n_seeds: int):
    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Matrix: {N_MODELS} x {N_BENCH}, observed={int(OBSERVED.sum())}")
    print(f"k values: {k_values}; seeds: {n_seeds}; shards: {len(family_keys) * len(k_values) * n_seeds}")
    for family_key in family_keys:
        landmark = landmark_by_key(family_key)
        target_ids = target_model_ids(landmark)
        train_ids = train_model_ids(landmark, target_ids)
        candidate_js = target_candidate_benchmarks(target_ids)
        print(
            f"  {landmark['family_name']:24s} cutoff={landmark['cutoff_date']} "
            f"train={len(train_ids):2d} target={len(target_ids):2d} "
            f"target_observed_union={len(candidate_js):3d}"
        )


def aggregate_metric(values: list[float]) -> dict:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"median": None, "iqr": None, "mean": None, "std": None, "values": []}
    return {
        "median": float(np.median(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "values": [float(v) for v in arr],
    }


def merge_results(family_keys: list[str], k_values: list[int], n_seeds: int):
    raw_predictions = []
    shard_metrics = []
    landmarks = {}
    seen_raw = set()
    for family_key, k, seed in expected_units(family_keys, k_values, n_seeds):
        path = shard_path(family_key, k, seed)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing shard: {path}")
        payload = load_json(path)
        validate_shard(payload, path, family_key, k, seed)
        cfg = payload["config"]
        landmarks[family_key] = {
            "family_name": cfg["family_name"],
            "cutoff_date": cfg["cutoff_date"],
            "target_model_ids": cfg["target_model_ids"],
            "train_model_ids": cfg["train_model_ids"],
            "n_target_models": len(cfg["target_model_ids"]),
            "n_train_models": len(cfg["train_model_ids"]),
        }
        shard_metrics.append({
            "family_key": family_key,
            "family_name": cfg["family_name"],
            "k": int(k),
            "seed": int(seed),
            **payload["metrics"],
            "n_eval_cells": int(cfg["n_eval_cells"]),
            "n_metric_cells": int(cfg["n_metric_cells"]),
            "n_revealed_cells": int(cfg["n_revealed_cells"]),
            "n_hidden_cells": int(cfg["n_hidden_cells"]),
            "n_not_predictable_cells": int(cfg["n_not_predictable_cells"]),
            "n_raw_predictions": int(cfg["n_raw_predictions"]),
            "active_target_model_ids": cfg["active_target_model_ids"],
            "revealed_by_model": cfg["revealed_by_model"],
        })
        for row in payload["raw_predictions"]:
            key = (
                row["family_key"],
                int(row["k"]),
                int(row["seed"]),
                row["model_id"],
                row["benchmark_id"],
            )
            if key in seen_raw:
                raise RuntimeError(f"Duplicate raw prediction key: {key}")
            seen_raw.add(key)
            raw_predictions.append(row)

    summary = {}
    for family_key in family_keys:
        summary[family_key] = {
            **landmarks[family_key],
            "by_k": {},
        }
        for k in k_values:
            rows = [
                row
                for row in shard_metrics
                if row["family_key"] == family_key and int(row["k"]) == int(k)
            ]
            summary[family_key]["by_k"][str(k)] = {
                "n_seeds": len(rows),
                "n_eval_cells_median": int(np.median([r["n_eval_cells"] for r in rows])) if rows else 0,
                "n_metric_cells_median": int(np.median([r["n_metric_cells"] for r in rows])) if rows else 0,
                "n_revealed_cells_median": int(np.median([r["n_revealed_cells"] for r in rows])) if rows else 0,
                "n_hidden_cells_median": int(np.median([r["n_hidden_cells"] for r in rows])) if rows else 0,
                "n_not_predictable_cells_median": int(np.median([r["n_not_predictable_cells"] for r in rows])) if rows else 0,
                "n_raw_predictions_median": int(np.median([r["n_raw_predictions"] for r in rows])) if rows else 0,
                "medape": aggregate_metric([r["medape"] for r in rows]),
                "medae": aggregate_metric([r["medae"] for r in rows]),
            }

    payload = {
        "config": {
            "protocol_version": PROTOCOL_VERSION,
            "base_seed": BASE_SEED,
            "k_values": k_values,
            "n_seeds": int(n_seeds),
            "family_keys": family_keys,
            "matrix_shape": [int(N_MODELS), int(N_BENCH)],
            "predictor": "predict_benchpress_scores",
            "evaluation_universe": "all observed cells for active target models are recorded; metrics use revealed cells plus hidden cells with finite BenchPress predictions",
            "aggregation": "paper-facing medians across seed-level MedAPE/MedAE",
        },
        "landmarks": [landmarks[key] | {"family_key": key} for key in family_keys],
        "summary_by_family": summary,
        "summary_by_family_k_seed": shard_metrics,
        "raw_predictions": sorted(
            raw_predictions,
            key=lambda row: (
                row["family_key"],
                int(row["k"]),
                int(row["seed"]),
                row["model_id"],
                row["benchmark_id"],
            ),
        ),
    }
    write_json(RESULTS_PATH, payload, indent=2, trailing_newline=True)
    print(f"Wrote {RESULTS_PATH} with {len(raw_predictions)} raw predictions")


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["describe", "shard", "run-all", "merge"], default="run-all")
    parser.add_argument("--family-key", choices=[x["family_key"] for x in LANDMARKS])
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--k-values", default=",".join(str(k) for k in K_VALUES))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--family-limit", type=int, default=None,
                        help="Debug/smoke only: keep the first N families.")
    args = parser.parse_args()

    np.random.seed(BASE_SEED)
    random.seed(BASE_SEED)

    k_values = parse_int_list(args.k_values)
    family_keys = [x["family_key"] for x in LANDMARKS]
    if args.family_limit is not None:
        family_keys = family_keys[: args.family_limit]

    if args.mode == "describe":
        describe_design(family_keys, k_values, args.n_seeds)
    elif args.mode == "shard":
        if args.family_key is None or args.k is None or args.seed is None:
            raise SystemExit("--mode shard requires --family-key, --k, and --seed")
        os.makedirs(SHARD_DIR, exist_ok=True)
        path = shard_path(args.family_key, args.k, args.seed)
        run_shard(args.family_key, args.k, args.seed, out_path=path)
        print(f"Wrote {path}")
    elif args.mode == "run-all":
        run_all(family_keys, k_values, args.n_seeds, args.workers)
    else:
        merge_results(family_keys, k_values, args.n_seeds)


if __name__ == "__main__":
    main()
