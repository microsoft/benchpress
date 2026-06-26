#!/usr/bin/env python
"""Build the public Hugging Face table export from the BenchPress JSON cache."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "benchpress" / "data" / "llm_benchmark_data.json"
DEFAULT_OUT_DIR = ROOT / "maintenance" / "exports" / "hf_dataset"
DEFAULT_REPO_ID = "microsoft/benchpress-score-matrix"
PUBLIC_SCHEMA_VERSION = "public-table-export-v1"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source JSON: {rel(path)}. Run `python -m benchpress.download_data` "
            "or place the audited matrix there first."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for key in ("models", "benchmarks", "scores"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"{rel(path)} must contain a `{key}[]` list")
    return data


def json_field(value: Any) -> str:
    if value in (None, ""):
        value = {}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json_field(value)


def models_table(models: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in models:
        rows.append(
            {
                "model_id": item.get("id"),
                "model_name": item.get("name"),
                "provider": item.get("provider"),
                "release_date": item.get("release_date"),
                "params_total_M": item.get("params_total_M"),
                "params_active_M": item.get("params_active_M"),
                "architecture": item.get("architecture"),
                "is_reasoning": item.get("is_reasoning"),
                "open_weights": item.get("open_weights"),
                "canonical_setting_json": json_field(item.get("canonical_setting")),
            }
        )
    return pd.DataFrame(rows)


def benchmarks_table(benchmarks: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in benchmarks:
        rows.append(
            {
                "benchmark_id": item.get("id"),
                "benchmark_name": item.get("name"),
                "category": item.get("category"),
                "metric": item.get("metric"),
                "num_problems": item.get("num_problems"),
                "source_url": item.get("source_url"),
                "canonical_setting_json": json_field(item.get("canonical_setting")),
            }
        )
    return pd.DataFrame(rows)


def scores_all_table(scores: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in scores:
        rows.append(
            {
                "model_id": item.get("model_id"),
                "benchmark_id": item.get("benchmark_id"),
                "score": item.get("score"),
                "reference_url": item.get("reference_url"),
                "reported_setting_json": json_field(item.get("reported_setting")),
                "matches_canonical": item.get("matches_canonical"),
                "source_type": item.get("source_type"),
                "audit_status": item.get("audit_status"),
                "notes": item.get("notes") or "",
            }
        )
    return pd.DataFrame(rows)


def paper_scores_table(raw: dict[str, Any], json_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    os.environ["BENCHPRESS_DATA"] = str(json_path)
    from benchpress.build_benchmark_matrix import load_score_matrix
    from benchpress.build_benchmark_matrix.build_benchmark_matrix import (
        BENCHMARK_FILL_RULES,
        DEFAULT_B_THRESHOLD,
        DEFAULT_EXCLUDED_BENCHMARKS,
        DEFAULT_EXCLUDED_MODELS,
        DEFAULT_M_THRESHOLD,
        MODEL_FILL_RULES,
        _apply_benchmark_fill_rules,
        _apply_canonical_rules,
        _apply_model_fill_rules,
        _filter_status,
        _iterate_threshold,
    )

    matrix, info = load_score_matrix(json_path=json_path, return_info=True)

    scores = _filter_status(raw["scores"], ("verified", "verified_third_party"))
    excluded_models = set(DEFAULT_EXCLUDED_MODELS)
    excluded_benchmarks = set(DEFAULT_EXCLUDED_BENCHMARKS)
    scores = _apply_benchmark_fill_rules(
        scores,
        fill_rules=BENCHMARK_FILL_RULES,
        exclude_models=excluded_models,
    )
    scores = _apply_model_fill_rules(
        scores,
        fill_rules=MODEL_FILL_RULES,
        exclude_benchmarks=excluded_benchmarks,
    )
    scores = _apply_canonical_rules(
        scores,
        exclude_models=excluded_models,
        exclude_benchmarks=excluded_benchmarks,
    )
    obs = [(s["model_id"], s["benchmark_id"]) for s in scores if s.get("score") is not None]
    keep_models, keep_benchmarks, _ = _iterate_threshold(
        obs, DEFAULT_M_THRESHOLD, DEFAULT_B_THRESHOLD
    )

    rows = []
    seen: set[tuple[str, str]] = set()
    for item in scores:
        key = (item.get("model_id"), item.get("benchmark_id"))
        if key in seen or key[0] not in keep_models or key[1] not in keep_benchmarks:
            continue
        if item.get("score") is None:
            continue
        seen.add(key)
        rows.append(
            {
                "model_id": key[0],
                "benchmark_id": key[1],
                "score": item.get("score"),
                "reference_url": item.get("reference_url"),
                "reported_setting_json": json_field(item.get("reported_setting")),
                "matches_canonical": item.get("matches_canonical"),
                "source_type": item.get("source_type"),
                "audit_status": item.get("audit_status"),
                "canonical_fill_from": item.get("_canonical_fill_from"),
                "notes": item.get("notes") or "",
            }
        )

    long_df = pd.DataFrame(rows)
    if len(long_df) != info.n_observations:
        raise RuntimeError(
            "paper long export does not match matrix observations: "
            f"{len(long_df)} rows vs {info.n_observations}"
        )
    return long_df, matrix.reset_index(), info


def parquet_available() -> bool:
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def write_table(df: pd.DataFrame, csv_path: Path, *, write_parquet: bool) -> list[str]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    written = [rel(csv_path)]
    if write_parquet:
        parquet_path = csv_path.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        written.append(rel(parquet_path))
    return written


def upload_folder(out_dir: Path, repo_id: str, commit_message: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Install `benchpress[hf]` or `huggingface_hub` before using --upload."
        ) from exc

    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out_dir),
        path_in_repo=".",
        commit_message=commit_message,
    )


def build_export(
    *,
    json_path: Path,
    out_dir: Path,
    parquet_mode: str,
    repo_id: str,
) -> dict[str, Any]:
    data = load_json(json_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if parquet_mode == "yes":
        write_parquet = True
        if not parquet_available():
            raise RuntimeError("Parquet export requested, but neither pyarrow nor fastparquet is installed.")
    elif parquet_mode == "no":
        write_parquet = False
    else:
        write_parquet = parquet_available()

    models_df = models_table(data["models"])
    benchmarks_df = benchmarks_table(data["benchmarks"])
    scores_all_df = scores_all_table(data["scores"])
    scores_paper_df, matrix_wide_df, info = paper_scores_table(data, json_path)

    files: list[str] = []
    files.extend(write_table(models_df, out_dir / "data" / "models.csv", write_parquet=write_parquet))
    files.extend(write_table(benchmarks_df, out_dir / "data" / "benchmarks.csv", write_parquet=write_parquet))
    files.extend(write_table(scores_all_df, out_dir / "data" / "scores_all.csv", write_parquet=write_parquet))
    files.extend(write_table(scores_paper_df, out_dir / "data" / "scores_paper.csv", write_parquet=write_parquet))
    files.append(rel(out_dir / "data" / "score_matrix_paper_wide.csv"))
    matrix_wide_df.to_csv(out_dir / "data" / "score_matrix_paper_wide.csv", index=False)

    metadata = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "dataset_id": repo_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_json": rel(json_path),
        "parquet_written": write_parquet,
        "rows": {
            "models": len(models_df),
            "benchmarks": len(benchmarks_df),
            "scores_all": len(scores_all_df),
            "scores_paper": len(scores_paper_df),
        },
        "paper_matrix": {
            "models": info.n_models,
            "benchmarks": info.n_benchmarks,
            "observations": info.n_observations,
            "fill_rate": info.fill_rate,
            "m_threshold": info.m_threshold,
            "b_threshold": info.b_threshold,
            "iterations": info.iterations,
        },
        "files": files,
        "notes": [
            "scores_all is the public pre-filter score table.",
            "scores_paper is the paper-canonical filtered long table.",
            "score_matrix_paper_wide.csv is the paper-canonical model-by-benchmark matrix.",
            "This public export may omit rich internal audit fields such as candidates[] and raw cost-evidence traces.",
        ],
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    files.append(rel(metadata_path))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BenchPress public HF dataset tables.")
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--parquet",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Write Parquet alongside CSV when possible (default: auto).",
    )
    parser.add_argument("--upload", action="store_true", help="Upload --out-dir to Hugging Face.")
    parser.add_argument(
        "--commit-message",
        default="Update BenchPress score matrix export",
        help="Hugging Face commit message used with --upload.",
    )
    args = parser.parse_args()

    metadata = build_export(
        json_path=args.json_path,
        out_dir=args.out_dir,
        parquet_mode=args.parquet,
        repo_id=args.repo_id,
    )
    print(f"Wrote {rel(args.out_dir)}")
    print(
        "Export summary: "
        f"{metadata['rows']['models']} models, "
        f"{metadata['rows']['benchmarks']} benchmarks, "
        f"{metadata['rows']['scores_all']} scores_all rows, "
        f"{metadata['paper_matrix']['models']}x{metadata['paper_matrix']['benchmarks']} paper matrix"
    )
    if args.parquet == "auto" and not metadata["parquet_written"]:
        print("Parquet skipped because pyarrow/fastparquet is not installed.", file=sys.stderr)
    if args.upload:
        upload_folder(args.out_dir, args.repo_id, args.commit_message)
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
