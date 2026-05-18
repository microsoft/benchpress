"""Build the canonical BenchPress default prediction artifacts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_NAMES,
    MODEL_IDS,
    MODEL_NAMES,
    compute_prediction_error,
)
from benchpress.io_utils import write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = REPO_ROOT / "experiments" / "sec4_building_benchpress" / "method_comparison"
DEFAULT_DIR = REPO_ROOT / "benchpress" / "evaluation" / "default_predictions" / "benchpress_default"


def _ensure_method_comparison() -> dict:
    results_path = METHOD_DIR / "results.json"
    manifest_path = METHOD_DIR / "manifest.json"
    if not results_path.exists() or not manifest_path.exists():
        subprocess.run([sys.executable, str(METHOD_DIR / "run.py"), "--merge"], check=True)
    with results_path.open() as f:
        results = json.load(f)
    with manifest_path.open() as f:
        manifest = json.load(f)
    if int(manifest.get("n_missing_shards", -1)) != 0:
        raise RuntimeError(
            "method-comparison manifest still has missing shards after --merge"
        )
    return results


def _default_prediction_file(results: dict) -> Path:
    row = results["logit"]["Bias ALS"]
    expected = {"rank": 2, "lam": 0.1}
    if row.get("best_hp") != expected:
        raise RuntimeError(
            f"BenchPress default recipe changed: expected {expected}, found {row.get('best_hp')}"
        )
    path = METHOD_DIR / row["prediction_file"]
    if not path.exists():
        subprocess.run([sys.executable, str(METHOD_DIR / "run.py"), "--merge"], check=True)
    if not path.exists():
        raise FileNotFoundError(f"missing default prediction shard: {path}")
    return path


def _raw_predictions_by(indices, fold_id, actual, predicted, other_indices):
    grouped = defaultdict(lambda: {
        "fold_id": [],
        "other_index": [],
        "actual": [],
        "predicted": [],
    })
    for group_idx, other_idx, fold, a, p in zip(indices, other_indices, fold_id, actual, predicted):
        bucket = grouped[int(group_idx)]
        bucket["fold_id"].append(int(fold))
        bucket["other_index"].append(int(other_idx))
        bucket["actual"].append(float(a))
        bucket["predicted"].append(float(p))
    return grouped


def _metric_row(actual, predicted) -> dict:
    metrics = compute_prediction_error(np.asarray(actual), np.asarray(predicted))
    return {
        "n": int(metrics["n"]),
        "medape": float(metrics["medape"]),
        "medae": float(metrics["medae"]),
    }


def build_default_predictions(output_dir: Path = DEFAULT_DIR) -> Path:
    """Create default ``predictions.npz`` / by-model / by-benchmark artifacts."""
    results = _ensure_method_comparison()
    source_path = _default_prediction_file(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(source_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    predictions_path = output_dir / "predictions.npz"
    if source_path.resolve() != predictions_path.resolve():
        shutil.copy2(source_path, predictions_path)

    fold_id = arrays["fold_id"].astype(int)
    test_i = arrays["test_i"].astype(int)
    test_j = arrays["test_j"].astype(int)
    actual = arrays["actual"].astype(float)
    predicted = arrays["predicted"].astype(float)

    setting = {
        "name": "benchpress_default",
        "source_prediction_file": os.path.relpath(source_path, REPO_ROOT),
        "predictor": "Logit + Bias ALS",
        "transform": "logit",
        "method": "Bias ALS",
        "rank": 2,
        "lambda": 0.1,
        "n_seeds": 10,
        "n_folds": 3,
        "base_seed": 42,
        "fold_source": "benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json",
        "n_predictions": int(len(actual)),
    }
    summary = _metric_row(actual, predicted)

    by_bench = []
    for j in sorted(set(test_j.tolist())):
        mask = test_j == j
        raw = {
            "fold_id": fold_id[mask].tolist(),
            "model_index": test_i[mask].tolist(),
            "actual": actual[mask].tolist(),
            "predicted": predicted[mask].tolist(),
        }
        by_bench.append({
            "bench_id": BENCH_IDS[j],
            "bench_name": BENCH_NAMES[BENCH_IDS[j]],
            "bench_index": int(j),
            **_metric_row(actual[mask], predicted[mask]),
            "raw_predictions": raw,
        })

    by_model = []
    for i in sorted(set(test_i.tolist())):
        mask = test_i == i
        raw = {
            "fold_id": fold_id[mask].tolist(),
            "benchmark_index": test_j[mask].tolist(),
            "actual": actual[mask].tolist(),
            "predicted": predicted[mask].tolist(),
        }
        by_model.append({
            "model_id": MODEL_IDS[i],
            "model_name": MODEL_NAMES[MODEL_IDS[i]],
            "model_index": int(i),
            **_metric_row(actual[mask], predicted[mask]),
            "raw_predictions": raw,
        })

    write_json_atomic(
        str(output_dir / "metadata.json"),
        {
            "description": "Canonical BenchPress default predictions derived from Section 4.2.",
            "setting": setting,
            "summary_metrics": summary,
        },
        indent=2,
        trailing_newline=True,
    )
    write_json_atomic(
        str(output_dir / "by_benchmark.json"),
        {"setting": setting, "benchmarks": by_bench},
        indent=2,
        trailing_newline=True,
    )
    write_json_atomic(
        str(output_dir / "by_model.json"),
        {"setting": setting, "models": by_model},
        indent=2,
        trailing_newline=True,
    )
    print(f"Wrote default BenchPress predictions to {output_dir}")
    return predictions_path


def main() -> None:
    build_default_predictions()


if __name__ == "__main__":
    main()
