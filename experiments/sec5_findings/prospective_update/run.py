#!/usr/bin/env python3
"""Prospective May-to-August BenchPress snapshot test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(HERE, "config.json")
DEFAULT_ORDER_PATH = os.path.abspath(os.path.join(
    HERE, "..", "optimal_probe", "all_known", "probe_orderings.json",
))
PROTOCOL = "prospective_update_may_to_august_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True, help="May 2026 score JSON.")
    parser.add_argument("--eval-json", required=True, help="August 26, 2026 score JSON.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-random-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--probe-order-json", default=DEFAULT_ORDER_PATH)
    return parser.parse_args()


def bootstrap_benchpress(train_json):
    os.environ["BENCHPRESS_DATA"] = os.path.abspath(os.path.expanduser(train_json))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from benchpress.build_benchmark_matrix import load_score_matrix
    from benchpress.evaluation_harness import (
        compute_prediction_error,
        make_score_predictor,
        matrix_identity_sha256,
    )
    from benchpress.io_utils import write_json_atomic
    from benchpress.methods.completers import complete_benchmark_mean, complete_model_mean
    from benchpress.methods.predictors import (
        predict_benchmark_median_scores,
        predict_benchpress_scores,
    )
    return {
        "load_score_matrix": load_score_matrix,
        "compute_prediction_error": compute_prediction_error,
        "make_score_predictor": make_score_predictor,
        "matrix_identity_sha256": matrix_identity_sha256,
        "write_json_atomic": write_json_atomic,
        "complete_benchmark_mean": complete_benchmark_mean,
        "complete_model_mean": complete_model_mean,
        "predict_benchmark_median_scores": predict_benchmark_median_scores,
        "predict_benchpress_scores": predict_benchpress_scores,
    }


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_snapshot(json_path, load_score_matrix):
    df, info, models_df, bench_df = load_score_matrix(
        json_path=os.path.abspath(os.path.expanduser(json_path)),
        return_info=True,
        return_metadata=True,
    )
    matrix = df.to_numpy(dtype=float)
    model_ids = [str(x) for x in df.index.tolist()]
    benchmark_ids = [str(x) for x in df.columns.tolist()]
    model_names = {
        str(idx): str(row.get("name", idx))
        for idx, row in models_df.iterrows()
    }
    benchmark_names = {
        str(idx): str(row.get("name", idx))
        for idx, row in bench_df.iterrows()
    }
    benchmark_categories = {
        str(idx): str(row.get("category", "Unknown") or "Unknown")
        for idx, row in bench_df.iterrows()
    }
    raw = load_json_file(os.path.abspath(os.path.expanduser(json_path)))
    raw_benchmarks = {str(row["id"]): row for row in raw["benchmarks"]}
    benchmark_metrics = {}
    for bid in benchmark_ids:
        setting = raw_benchmarks.get(bid, {}).get("canonical_setting") or {}
        benchmark_metrics[bid] = {
            "type": setting.get("metric_type"),
            "range": setting.get("range"),
            "higher_is_better": setting.get("higher_is_better", True),
        }
    return {
        "df": df,
        "info": info,
        "matrix": matrix,
        "model_ids": model_ids,
        "benchmark_ids": benchmark_ids,
        "model_names": model_names,
        "benchmark_names": benchmark_names,
        "benchmark_categories": benchmark_categories,
        "benchmark_metrics": benchmark_metrics,
    }


def finite_or_none(value):
    return float(value) if value is not None and np.isfinite(float(value)) else None


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def summarize_rows(rows, compute_prediction_error):
    actual = np.array([row["actual"] for row in rows], dtype=float)
    pred = np.array([
        np.nan if row.get("pred") is None else row["pred"]
        for row in rows
    ], dtype=float)
    metrics = compute_prediction_error(actual, pred, aggregation="pool")
    finite = np.isfinite(actual) & np.isfinite(pred)
    abs_err = np.abs(pred[finite] - actual[finite])
    return {
        "n": int(metrics["n"]),
        "total_cells": int(len(rows)),
        "coverage": float(metrics["n"]) / float(len(rows)) if rows else None,
        "medae": finite_or_none(metrics["medae"]),
        "medape": finite_or_none(metrics["medape"]),
        "p90_abs_error": float(np.percentile(abs_err, 90)) if len(abs_err) else None,
    }


def grouped_summary(rows, group_fields, compute_prediction_error):
    groups = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups, key=lambda item: tuple("" if x is None else str(x) for x in item)):
        entry = {field: value for field, value in zip(group_fields, key)}
        entry.update(summarize_rows(groups[key], compute_prediction_error))
        output.append(entry)
    return output


def random_order(benchmark_ids, seed_idx, base_seed):
    rng = np.random.RandomState((int(base_seed) + int(seed_idx)) * 100000)
    perm = np.arange(len(benchmark_ids))
    rng.shuffle(perm)
    return [benchmark_ids[int(i)] for i in perm]


def benchmark_category_counts(rows):
    counts = {}
    for row in rows:
        category = row["benchmark_category"]
        entry = counts.setdefault(category, {
            "category": category,
            "cells": 0,
            "finite_benchpress_predictions": 0,
        })
        entry["cells"] += 1
        if row["method"] == "benchpress" and row.get("pred") is not None:
            entry["finite_benchpress_predictions"] += 1
    return sorted(counts.values(), key=lambda row: row["category"])


def part_i_new_cells(may, aug, bp, out):
    compute_prediction_error = bp["compute_prediction_error"]
    make_score_predictor = bp["make_score_predictor"]
    complete_benchmark_mean = bp["complete_benchmark_mean"]
    complete_model_mean = bp["complete_model_mean"]
    predict_benchpress_scores = bp["predict_benchpress_scores"]

    common_models = [mid for mid in may["model_ids"] if mid in aug["df"].index]
    common_benchmarks = [bid for bid in may["benchmark_ids"] if bid in aug["df"].columns]
    may_model_idx = {mid: idx for idx, mid in enumerate(may["model_ids"])}
    may_bench_idx = {bid: idx for idx, bid in enumerate(may["benchmark_ids"])}
    new_cells = []
    common_pairs = []
    for mid in common_models:
        for bid in common_benchmarks:
            may_value = may["df"].at[mid, bid]
            aug_value = aug["df"].at[mid, bid]
            if np.isfinite(aug_value) and not np.isfinite(may_value):
                new_cells.append((mid, bid, float(aug_value)))
            if np.isfinite(aug_value) and np.isfinite(may_value):
                common_pairs.append((float(may_value), float(aug_value)))

    predictors = {
        "benchpress": predict_benchpress_scores(
            may["matrix"],
            metric=may["benchmark_metrics"],
            benchmark_ids=may["benchmark_ids"],
        ),
        "logit_benchmark_mean": make_score_predictor(
            complete_benchmark_mean,
            "logit",
            metric=may["benchmark_metrics"],
            benchmark_ids=may["benchmark_ids"],
        )(may["matrix"]),
        "logit_model_mean": make_score_predictor(
            complete_model_mean,
            "logit",
            metric=may["benchmark_metrics"],
            benchmark_ids=may["benchmark_ids"],
        )(may["matrix"]),
    }

    raw = []
    for method, pred_matrix in predictors.items():
        for mid, bid, actual in new_cells:
            i = may_model_idx[mid]
            j = may_bench_idx[bid]
            raw.append({
                "part": "new_cells_existing_models",
                "method": method,
                "model_id": mid,
                "model_name": may["model_names"].get(mid, mid),
                "benchmark_id": bid,
                "benchmark_name": may["benchmark_names"].get(bid, bid),
                "benchmark_category": may["benchmark_categories"].get(bid, "Unknown"),
                "actual": actual,
                "pred": finite_or_none(pred_matrix[i, j]),
                "k": None,
                "seed": None,
                "ordering": None,
                "is_probe": False,
            })

    by_method = grouped_summary(
        raw,
        ["method"],
        compute_prediction_error,
    )
    common_may = np.array([row[0] for row in common_pairs], dtype=float)
    common_aug = np.array([row[1] for row in common_pairs], dtype=float)
    common_metrics = compute_prediction_error(common_may, common_aug, aggregation="pool")
    out["part_i"] = {
        "summary_by_method": by_method,
        "category_breakdown_counts": benchmark_category_counts([
            row for row in raw if row["method"] == "benchpress"
        ]),
        "snapshot_overlap_sanity": {
            "n_common_cells": int(len(common_pairs)),
            "fraction_identical": float(np.mean(common_may == common_aug)) if len(common_pairs) else None,
            "medae_between_versions": finite_or_none(common_metrics["medae"]),
        },
        "n_new_cells": int(len(new_cells)),
    }
    return raw


def observed_aug_values_for_may_benchmarks(aug, model_id, may_benchmark_ids):
    row = aug["df"].loc[model_id]
    values = {}
    for bid in may_benchmark_ids:
        if bid in row.index and np.isfinite(row[bid]):
            values[bid] = float(row[bid])
    return values


def append_target_and_predict(may, aug_values, probe_ids, predict_benchpress_scores):
    target_row = np.full((1, len(may["benchmark_ids"])), np.nan, dtype=float)
    bench_idx = {bid: idx for idx, bid in enumerate(may["benchmark_ids"])}
    for bid in probe_ids:
        if bid in aug_values and bid in bench_idx:
            target_row[0, bench_idx[bid]] = aug_values[bid]
    train = np.vstack([may["matrix"], target_row])
    pred = predict_benchpress_scores(
        train,
        metric=may["benchmark_metrics"],
        benchmark_ids=may["benchmark_ids"],
    )
    return pred[-1, :]


def add_part_ii_rows(raw, may, aug, model_id, method, k, ordering, seed,
                     prefix, predict_row):
    observed_values = observed_aug_values_for_may_benchmarks(
        aug, model_id, may["benchmark_ids"],
    )
    prefix_set = set(prefix)
    hidden_ids = [bid for bid in may["benchmark_ids"] if bid in observed_values and bid not in prefix_set]
    bench_idx = {bid: idx for idx, bid in enumerate(may["benchmark_ids"])}
    n_observed_probe = sum(1 for bid in prefix if bid in observed_values)
    for bid in hidden_ids:
        raw.append({
            "part": "new_models_frozen_probe_set",
            "method": method,
            "model_id": model_id,
            "model_name": aug["model_names"].get(model_id, model_id),
            "benchmark_id": bid,
            "benchmark_name": may["benchmark_names"].get(bid, bid),
            "benchmark_category": may["benchmark_categories"].get(bid, "Unknown"),
            "actual": observed_values[bid],
            "pred": finite_or_none(predict_row[bench_idx[bid]]),
            "k": int(k),
            "seed": seed,
            "ordering": ordering,
            "is_probe": False,
            "n_observed_probe": int(n_observed_probe),
            "n_hidden_observed": int(len(hidden_ids)),
        })


def part_ii_new_models(may, aug, bp, args, orderings, out):
    predict_benchpress_scores = bp["predict_benchpress_scores"]
    predict_benchmark_median_scores = bp["predict_benchmark_median_scores"]
    compute_prediction_error = bp["compute_prediction_error"]
    k_values = [1, 3, 5, 10]
    new_models = [mid for mid in aug["model_ids"] if mid not in set(may["model_ids"])]
    raw = []

    median_pred = predict_benchmark_median_scores(may["matrix"])
    median_row = median_pred[0, :]
    for mid in new_models:
        observed_values = observed_aug_values_for_may_benchmarks(aug, mid, may["benchmark_ids"])
        if len(observed_values) >= 5:
            add_part_ii_rows(
                raw, may, aug, mid, "benchmark_median_k0", 0,
                "benchmark_median", None, [], median_row,
            )

    for ordering_key in ("medae_any", "medae_low_cost"):
        order_ids = orderings[ordering_key]["benchmark_ids"]
        for k in k_values:
            prefix = order_ids[:k]
            for mid in new_models:
                observed_values = observed_aug_values_for_may_benchmarks(
                    aug, mid, may["benchmark_ids"],
                )
                n_probe = sum(1 for bid in prefix if bid in observed_values)
                n_hidden = sum(1 for bid in observed_values if bid not in set(prefix))
                if n_probe < 1 or n_hidden < 5:
                    continue
                pred_row = append_target_and_predict(
                    may, observed_values, prefix, predict_benchpress_scores,
                )
                add_part_ii_rows(
                    raw, may, aug, mid, "benchpress_fixed_order", k,
                    ordering_key, None, prefix, pred_row,
                )

    for seed_idx in range(int(args.n_random_seeds)):
        order_ids = random_order(may["benchmark_ids"], seed_idx, args.base_seed)
        for k in k_values:
            prefix = order_ids[:k]
            prefix_set = set(prefix)
            for mid in new_models:
                observed_values = observed_aug_values_for_may_benchmarks(
                    aug, mid, may["benchmark_ids"],
                )
                n_probe = sum(1 for bid in prefix if bid in observed_values)
                n_hidden = sum(1 for bid in observed_values if bid not in prefix_set)
                if n_probe < 1 or n_hidden < 5:
                    continue
                pred_row = append_target_and_predict(
                    may, observed_values, prefix, predict_benchpress_scores,
                )
                add_part_ii_rows(
                    raw, may, aug, mid, "benchpress_random_order", k,
                    "random", int(seed_idx), prefix, pred_row,
                )

    part_rows = [row for row in raw if row["part"] == "new_models_frozen_probe_set"]
    out["part_ii"] = {
        "summary_pooled": grouped_summary(
            part_rows,
            ["method", "ordering", "k"],
            compute_prediction_error,
        ),
        "summary_by_seed": grouped_summary(
            [row for row in part_rows if row["seed"] is not None],
            ["method", "ordering", "k", "seed"],
            compute_prediction_error,
        ),
        "summary_per_target": grouped_summary(
            part_rows,
            ["method", "ordering", "k", "model_id"],
            compute_prediction_error,
        ),
        "new_model_count": int(len(new_models)),
    }
    return raw


def table_summary(summary):
    return {
        "part_i_new_cells_existing_models": summary["part_i"]["summary_by_method"],
        "part_i_category_counts": summary["part_i"]["category_breakdown_counts"],
        "part_ii_new_models_pooled": summary["part_ii"]["summary_pooled"],
        "part_ii_new_models_by_seed": summary["part_ii"]["summary_by_seed"],
        "part_ii_new_models_per_target": summary["part_ii"]["summary_per_target"],
    }


def main():
    args = parse_args()
    cfg = load_json_file(DEFAULT_CONFIG_PATH)
    bp = bootstrap_benchpress(args.train_json)
    os.makedirs(args.out_dir, exist_ok=True)

    may = load_snapshot(args.train_json, bp["load_score_matrix"])
    aug = load_snapshot(args.eval_json, bp["load_score_matrix"])
    may_observed = int(np.isfinite(may["matrix"]).sum())
    if list(may["matrix"].shape) != cfg["expected_train_shape"] or may_observed != int(cfg["expected_train_observed"]):
        raise RuntimeError(
            "May train matrix identity check failed: expected "
            f"{cfg['expected_train_shape']} / {cfg['expected_train_observed']} observed, "
            f"found {list(may['matrix'].shape)} / {may_observed}"
        )

    order_payload = load_json_file(args.probe_order_json)
    orderings = order_payload["orderings"]
    missing = [
        bid for key in ("medae_any", "medae_low_cost")
        for bid in orderings[key]["benchmark_ids"]
        if bid not in may["benchmark_ids"]
    ]
    if missing:
        raise RuntimeError(f"Probe ordering ids missing from May matrix: {sorted(set(missing))}")

    manifest = {
        "git_commit": git_commit(),
        "train_json": os.path.abspath(os.path.expanduser(args.train_json)),
        "eval_json": os.path.abspath(os.path.expanduser(args.eval_json)),
        "train_matrix_identity_sha256": bp["matrix_identity_sha256"](may["matrix"]),
        "eval_matrix_identity_sha256": bp["matrix_identity_sha256"](aug["matrix"]),
        "train_shape": [int(x) for x in may["matrix"].shape],
        "eval_shape": [int(x) for x in aug["matrix"].shape],
        "train_n_observed": may_observed,
        "eval_n_observed": int(np.isfinite(aug["matrix"]).sum()),
    }
    config = {
        "protocol": PROTOCOL,
        "k_values": cfg["k_values"],
        "n_random_seeds": int(args.n_random_seeds),
        "base_seed": int(args.base_seed),
        "probe_order_json": os.path.relpath(
            os.path.abspath(os.path.expanduser(args.probe_order_json)),
            REPO_ROOT,
        ),
        "probe_order_source": order_payload.get("source"),
        "predictor": "predict_benchpress_scores (Logit Bias ALS, rank=2, lambda=0.1)",
        "part_i": "Predict August-observed cells for May models and May benchmarks that were missing in May.",
        "part_ii": "Append one August-only target model at a time to the May matrix with only probe cells revealed.",
        "inclusion_rule_part_ii": "k>0 requires >=1 observed probe cell in the prefix and >=5 hidden observed May-benchmark cells; k=0 benchmark-median baseline requires >=5 observed May-benchmark cells.",
    }

    summary = {
        "config": config,
        "manifest": manifest,
    }
    raw = []
    raw.extend(part_i_new_cells(may, aug, bp, summary))
    raw.extend(part_ii_new_models(may, aug, bp, args, orderings, summary))

    write_json_atomic = bp["write_json_atomic"]
    write_json_atomic(os.path.join(args.out_dir, "manifest.json"), manifest, indent=2)
    write_json_atomic(os.path.join(args.out_dir, "config.json"), config, indent=2)
    write_json_atomic(os.path.join(args.out_dir, "raw_predictions.json.gz"), raw)
    write_json_atomic(os.path.join(args.out_dir, "summary.json"), summary, indent=2)
    write_json_atomic(
        os.path.join(args.out_dir, "table_summary.json"),
        table_summary(summary),
        indent=2,
    )
    print(f"Wrote prospective update results under {args.out_dir}")


if __name__ == "__main__":
    main()
