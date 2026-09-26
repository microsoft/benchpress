#!/usr/bin/env python3
"""Figure 1 panel A: target-hidden keep-k error curves for selected cells.

For each target cell (model, benchmark) and each (k, seed), the target model's
row keeps k scores sampled from its other observed benchmarks and hides the
rest, including the target cell. BenchPress is fit on the masked matrix and the
absolute error on the target cell is recorded. The k=0 point is the benchmark
median of the target column over the other models.

A target model below the paper row threshold is appended to the thresholded
matrix as an extra row holding its observed scores on the matrix benchmarks.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROTOCOL = "target_hidden_keep_k"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", required=True, help="Score matrix JSON snapshot.")
    parser.add_argument("--target", action="append", required=True,
                        help="MODEL_ID:BENCH_ID; repeat for several target cells.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    score_json = os.path.abspath(os.path.expanduser(args.score_json))
    # The harness builds its thresholded matrix from BENCHPRESS_DATA at import time.
    os.environ["BENCHPRESS_DATA"] = score_json
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from benchpress.build_benchmark_matrix import load_score_matrix
    from benchpress.evaluation_harness import (
        BENCH_IDS, BENCH_NAMES, M_FULL, MODEL_IDS, matrix_identity_sha256,
    )
    from benchpress.io_utils import load_json, write_json_atomic
    from benchpress.methods.predictors import (
        predict_benchmark_median_scores, predict_benchpress_scores,
    )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    config = {
        "protocol": PROTOCOL,
        "commit": commit,
        "score_json": os.path.basename(score_json),
        "matrix_shape": list(M_FULL.shape),
        "matrix_n_observed": int(np.isfinite(M_FULL).sum()),
        "matrix_identity_sha256": matrix_identity_sha256(M_FULL),
        "k_values": list(range(1, args.k_max + 1)),
        "seeds": list(range(args.n_seeds)),
        "base_seed": args.base_seed,
        "predictor": "predict_benchpress_scores",
        "baseline": "predict_benchmark_median_scores",
    }
    payload = {"config": config, "targets": {}}
    if os.path.exists(args.output):
        payload = load_json(args.output)
        if payload["config"] != config:
            raise RuntimeError(f"{args.output} has a different config; delete it to rerun.")

    full_scores = load_score_matrix(
        json_path=score_json, m_threshold=0, b_threshold=0, deduplicate=True,
    ).reindex(columns=BENCH_IDS)
    model_names = {m["id"]: m.get("name") for m in load_json(score_json)["models"]}
    for target in args.target:
        if target in payload["targets"]:
            print(f"{target}: done, skipped", flush=True)
            continue
        model_id, bench_id = target.split(":")
        j = BENCH_IDS.index(bench_id)
        if model_id in MODEL_IDS:
            base, row = M_FULL, MODEL_IDS.index(model_id)
        else:
            base = np.vstack([M_FULL, full_scores.loc[model_id].to_numpy(dtype=float)])
            row = base.shape[0] - 1
        actual = float(base[row, j])
        others = np.array([c for c in np.where(np.isfinite(base[row]))[0] if c != j])
        if not np.isfinite(actual) or len(others) < args.k_max:
            raise RuntimeError(f"{target}: target unobserved or fewer than k-max other cells.")

        hidden_row = base.copy()
        hidden_row[row] = np.nan
        baseline_pred = float(predict_benchmark_median_scores(hidden_row)[row, j])
        raw = []
        for k in config["k_values"]:
            for seed_idx in config["seeds"]:
                rng = np.random.RandomState(
                    (args.base_seed + seed_idx) * 100000 + k * 1000 + row)
                keep = rng.permutation(others)[:k]
                m_train = hidden_row.copy()
                m_train[row, keep] = base[row, keep]
                pred = float(predict_benchpress_scores(m_train)[row, j])
                raw.append({
                    "k": k, "seed": seed_idx,
                    "revealed": [BENCH_IDS[c] for c in sorted(keep)],
                    "pred": pred, "abs_error": abs(pred - actual),
                })
            print(f"{target}: k={k} done", flush=True)
        payload["targets"][target] = {
            "model_id": model_id,
            "bench_id": bench_id,
            "model": model_names[model_id],
            "benchmark": BENCH_NAMES[bench_id],
            "in_thresholded_matrix": model_id in MODEL_IDS,
            "n_other_observed": int(len(others)),
            "actual": actual,
            "baseline_pred": baseline_pred,
            "baseline_ae": abs(baseline_pred - actual),
            "random": [
                {
                    "k": k,
                    "median": float(np.median(errs)),
                    "q1": float(np.percentile(errs, 25)),
                    "q3": float(np.percentile(errs, 75)),
                }
                for k in config["k_values"]
                for errs in [[r["abs_error"] for r in raw if r["k"] == k]]
            ],
            "raw": raw,
        }
        write_json_atomic(args.output, payload, indent=1)
        print(f"{target}: wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
