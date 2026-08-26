#!/usr/bin/env python
"""Build a filtered, predicted browser artifact from a score-matrix CSV."""

import argparse
import csv
import gzip
import hashlib
import json
import os

import numpy as np

from benchpress.methods.predictors import predict_benchpress_scores


def load_score_matrix(scores_path):
    with open(scores_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[0].strip().lstrip("\ufeff") != "model":
            raise ValueError("first CSV header cell must be 'model'")
        benchmark_ids = [value.strip() for value in header[1:]]
        if not all(benchmark_ids) or len(set(benchmark_ids)) != len(benchmark_ids):
            raise ValueError("benchmark ids must be non-empty and unique")

        model_ids = []
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            if len(row) > len(header):
                raise ValueError(
                    f"CSV row {line_number} has {len(row)} cells; "
                    f"expected {len(header)}")
            row += [""] * (len(header) - len(row))
            model_id = row[0].strip()
            if not model_id:
                raise ValueError(f"empty model id on CSV row {line_number}")
            model_ids.append(model_id)
            rows.append([
                np.nan if not cell.strip() else float(cell.strip().rstrip("%"))
                for cell in row[1:]
            ])
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model ids must be unique")

    meta_path = os.path.splitext(scores_path)[0] + ".meta.json"
    with open(meta_path) as f:
        metric = json.load(f)
    return model_ids, benchmark_ids, np.asarray(rows, dtype=float), metric


def filter_percentage_matrix(
        model_ids, benchmark_ids, values, metric,
        min_benchmarks_per_model, min_models_per_benchmark):
    pct_columns = np.asarray([
        metric.get(benchmark_id, {}).get("type", "pct") == "pct"
        for benchmark_id in benchmark_ids
    ])
    benchmark_ids = [
        benchmark_id for benchmark_id, keep
        in zip(benchmark_ids, pct_columns, strict=True) if keep
    ]
    values = values[:, pct_columns]

    row_keep = np.ones(values.shape[0], dtype=bool)
    col_keep = np.ones(values.shape[1], dtype=bool)
    while True:
        sub = values[np.ix_(row_keep, col_keep)]
        next_rows = row_keep.copy()
        next_cols = col_keep.copy()
        next_rows[np.where(row_keep)[0]] = (
            np.isfinite(sub).sum(axis=1) >= min_benchmarks_per_model)
        sub = values[np.ix_(next_rows, col_keep)]
        next_cols[np.where(col_keep)[0]] = (
            np.isfinite(sub).sum(axis=0) >= min_models_per_benchmark)
        if np.array_equal(next_rows, row_keep) and np.array_equal(
                next_cols, col_keep):
            break
        row_keep, col_keep = next_rows, next_cols
    if not row_keep.any() or not col_keep.any():
        raise ValueError("observation filtering removed the entire matrix")
    return (
        [model_id for model_id, keep
         in zip(model_ids, row_keep, strict=True) if keep],
        [benchmark_id for benchmark_id, keep
         in zip(benchmark_ids, col_keep, strict=True) if keep],
        values[np.ix_(row_keep, col_keep)],
    )


def clean_matrix(values):
    return [
        [None if not np.isfinite(value) else round(float(value), 3)
         for value in row]
        for row in values
    ]


def build_artifact(args):
    model_ids, benchmark_ids, values, metric = load_score_matrix(args.scores)
    model_ids, benchmark_ids, values = filter_percentage_matrix(
        model_ids,
        benchmark_ids,
        values,
        metric,
        args.min_benchmarks_per_model,
        args.min_models_per_benchmark,
    )
    predicted = predict_benchpress_scores(values)
    with open(args.scores, "rb") as f:
        input_sha256 = hashlib.sha256(f.read()).hexdigest()
    return {
        "meta": {
            "id": args.id,
            "name": args.name,
            "source_url": args.source_url,
            "snapshot_date": args.snapshot_date,
            "source_license": args.source_license,
            "models": len(model_ids),
            "benchmarks": len(benchmark_ids),
            "observed_cells": int(np.isfinite(values).sum()),
            "input_sha256": input_sha256,
            "filter": {
                "metric_type": "pct",
                "min_benchmarks_per_model": args.min_benchmarks_per_model,
                "min_models_per_benchmark": args.min_models_per_benchmark,
                "fixed_point": True,
            },
            "prediction_method": "Logit Bias ALS",
            "rank": 2,
            "lambda": 0.1,
            "confidence_available": False,
        },
        "models": [
            {"id": model_id, "name": model_id, "provider": args.name}
            for model_id in model_ids
        ],
        "benchmarks": [
            {"id": benchmark_id, "name": benchmark_id,
             "metric": {"type": "pct", "range": [0, 100],
                        "higher_is_better": True}}
            for benchmark_id in benchmark_ids
        ],
        "observed": clean_matrix(values),
        "predictions": clean_matrix(predicted),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-benchmarks-per-model", type=int, default=15)
    parser.add_argument("--min-models-per-benchmark", type=int, default=8)
    args = parser.parse_args()

    payload = json.dumps(
        build_artifact(args), separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    output = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as f:
        f.write(gzip.compress(payload, compresslevel=9, mtime=0))
    print(f"wrote {output} ({len(payload)} bytes before gzip)")


if __name__ == "__main__":
    main()
