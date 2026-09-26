#!/usr/bin/env python3
"""Evaluate a fixed all-known probe ordering without rerunning greedy search."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_NAMES,
    M_FULL,
    MODEL_IDS,
    MODEL_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    compute_prediction_error,
    evaluate_probe_set,
    matrix_identity_sha256,
)
from benchpress.io_utils import load_json, write_json_atomic
from benchpress.methods.predictors import predict_benchpress_scores

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_ORDER_JSON = os.path.join(SCRIPT_DIR, "probe_orderings.json")
PROTOCOL = "fixed_order_all_known_hidden_only_v1"


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def _manifest():
    return {
        "git_commit": _git_commit(),
        "matrix_identity_sha256": matrix_identity_sha256(M_FULL),
        "matrix_shape": [int(N_MODELS), int(N_BENCH)],
        "n_observed": int(OBSERVED.sum()),
    }


def _error_summary(rows, total_cells):
    actual = np.array([row["actual"] for row in rows], dtype=float)
    predicted = np.array([
        np.nan if row["pred"] is None else row["pred"]
        for row in rows
    ], dtype=float)
    metrics = compute_prediction_error(actual, predicted, aggregation="pool")
    finite = np.isfinite(actual) & np.isfinite(predicted)
    abs_err = np.abs(predicted[finite] - actual[finite])
    return {
        "n": int(metrics["n"]),
        "total_cells": int(total_cells),
        "coverage": (float(metrics["n"]) / float(total_cells)) if total_cells else None,
        "medae": _finite_or_none(metrics["medae"]),
        "medape": _finite_or_none(metrics["medape"]),
        "p90_abs_error": float(np.percentile(abs_err, 90)) if len(abs_err) else None,
    }


def _evaluate_k(args):
    k, order_indices = args
    prefix = order_indices[:k]
    probe_set = set(prefix)
    predictions, _, _ = evaluate_probe_set(
        prefix,
        predict_benchpress_scores,
        metric="medae",
    )
    rows = []
    for i, j, actual, pred in sorted(predictions, key=lambda row: (row[0], row[1])):
        is_revealed = int(j) in probe_set
        rows.append({
            "k": int(k),
            "target_model_index": int(i),
            "target_model_id": MODEL_IDS[int(i)],
            "target_model_name": MODEL_NAMES.get(MODEL_IDS[int(i)], MODEL_IDS[int(i)]),
            "benchmark_index": int(j),
            "benchmark_id": BENCH_IDS[int(j)],
            "benchmark_name": BENCH_NAMES.get(BENCH_IDS[int(j)], BENCH_IDS[int(j)]),
            "actual": float(actual),
            "pred": _finite_or_none(pred),
            "is_revealed": bool(is_revealed),
            "prediction_source": "revealed" if is_revealed else "benchpress",
        })
    hidden_rows = [row for row in rows if not row["is_revealed"]]
    return {
        "k": int(k),
        "probe_ids": [BENCH_IDS[int(j)] for j in prefix],
        "with_probe_zero": _error_summary(rows, len(rows)),
        "hidden_only": _error_summary(hidden_rows, len(hidden_rows)),
        "raw_predictions": rows,
    }


def _load_order(path, key):
    payload = load_json(path)
    orderings = payload.get("orderings", {})
    if key not in orderings:
        raise SystemExit(
            f"Unknown --fixed-order {key!r}; available: {', '.join(sorted(orderings))}"
        )
    order = orderings[key]
    missing = [bid for bid in order["benchmark_ids"] if bid not in BENCH_IDS]
    if missing:
        raise SystemExit(f"Ordering {key!r} contains benchmark ids not in this matrix: {missing}")
    return payload, order


def _resolve_out_path(out_arg, fixed_order):
    if out_arg is None:
        out_arg = f"fixed_order_{fixed_order}_hidden_only.json.gz"
    if os.path.isabs(out_arg):
        return out_arg
    return os.path.join(RESULTS_DIR, out_arg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-order-json", default=DEFAULT_ORDER_JSON)
    parser.add_argument("--fixed-order", required=True,
                        help="Key in --fixed-order-json, e.g. medae_any or medae_low_cost.")
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    order_payload, order = _load_order(args.fixed_order_json, args.fixed_order)
    k_max = len(order["benchmark_ids"]) if args.k_max is None else int(args.k_max)
    if k_max < 1 or k_max > len(order["benchmark_ids"]):
        raise SystemExit(f"--k-max must be in [1, {len(order['benchmark_ids'])}]")

    order_indices = [BENCH_IDS.index(bid) for bid in order["benchmark_ids"]]
    units = [(k, order_indices) for k in range(1, k_max + 1)]
    if args.workers <= 1:
        evaluated = [_evaluate_k(unit) for unit in units]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(units))) as pool:
            evaluated = list(pool.map(_evaluate_k, units))
    evaluated.sort(key=lambda row: row["k"])

    raw_predictions = []
    for row in evaluated:
        raw_predictions.extend(row.pop("raw_predictions"))

    output = {
        "config": {
            "protocol": PROTOCOL,
            "fixed_order_key": args.fixed_order,
            "fixed_order_json": os.path.relpath(
                os.path.abspath(os.path.expanduser(args.fixed_order_json)),
                REPO_ROOT,
            ),
            "k_max": int(k_max),
            "seeds": [],
            "order_label": order.get("label"),
            "candidate_pool": order.get("candidate_pool"),
            "benchmark_ids": order["benchmark_ids"],
            "display_names": order.get("display_names"),
            "order_source": order_payload.get("source"),
            "n_models": int(N_MODELS),
            "n_bench": int(N_BENCH),
            "n_observed": int(OBSERVED.sum()),
            "prediction_engine": "predict_benchpress_scores (Logit Bias ALS, rank=2, lambda=0.1)",
            "eval_scope": "all observed cells; probe cells are exact for with_probe_zero and excluded for hidden_only",
        },
        "manifest": _manifest(),
        "summary_by_k": evaluated,
        "raw_predictions": raw_predictions,
    }

    out_path = _resolve_out_path(args.out, args.fixed_order)
    write_json_atomic(out_path, output, indent=2)
    print(f"Saved -> {out_path}")
    for row in output["summary_by_k"]:
        if row["k"] in (5, 10):
            print(
                f"k={row['k']:2d} with_probe_zero MedAE={row['with_probe_zero']['medae']:.3f} "
                f"hidden_only MedAE={row['hidden_only']['medae']:.3f} "
                f"coverage={row['hidden_only']['coverage']:.3f}"
            )


if __name__ == "__main__":
    main()
