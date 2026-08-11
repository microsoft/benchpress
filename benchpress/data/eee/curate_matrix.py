#!/usr/bin/env python
"""Curate EEE aggregate results into a BenchPress-compatible score matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

repo_id = "evaleval/EEE_datastore"


metric_pct_names = {
    "accuracy",
    "acc",
    "exact_match",
    "em",
    "f1",
    "f1_macro",
    "auroc",
    "auc",
    "pass_rate",
    "pass_at_k",
    "pass@1",
    "win_rate",
    "recall",
    "precision",
}


def clean_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"infinity", "+infinity", "-infinity", "nan"}:
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def slug(value: object) -> str:
    text = str(value or "continuous").strip().lower()
    text = re.sub(r"[^a-z0-9@.+_-]+", "_", text)
    return text.strip("_") or "continuous"


def eee_aggregate_paths(api: HfApi, benchmarks: set[str] | None) -> list[str]:
    info = api.dataset_info(repo_id)
    paths = []
    for sibling in info.siblings:
        path = sibling.rfilename
        if not path.startswith("data/"):
            continue
        if not path.endswith(".json") or path.endswith("_samples.json"):
            continue
        benchmark = path.split("/", 2)[1]
        if benchmarks is not None and benchmark not in benchmarks:
            continue
        paths.append(path)
    return sorted(paths)


def download_aggregate(path: str, output: Path, force: bool) -> Path:
    target = output / "raw" / "eee" / path
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=path,
        token=True,
        force_download=force,
    )
    source = Path(cached)
    if source.resolve() != target.resolve():
        target.write_bytes(source.read_bytes())
    return target


def download_aggregates(paths: list[str], output: Path, force: bool, workers: int) -> list[tuple[str, str]]:
    pending = [path for path in paths if force or not (output / "raw" / "eee" / path).exists()]
    failures: list[tuple[str, str]] = []
    if not pending:
        return failures

    def download_with_retries(path: str) -> tuple[str, str] | None:
        last_error = ""
        for _ in range(3):
            try:
                download_aggregate(path, output, force)
                return None
            except Exception as exc:
                last_error = str(exc)
        return path, last_error

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_with_retries, path) for path in pending]
        for future in as_completed(futures):
            done += 1
            failure = future.result()
            if failure is not None:
                failures.append(failure)
            if done % 500 == 0 or done == len(pending):
                print(f"EEE download progress: {done}/{len(pending)} pending files")
    return failures


def local_aggregate_paths(output: Path) -> list[Path]:
    root = output / "raw" / "eee" / "data"
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*.json")
        if not path.name.endswith("_samples.json")
    )


def parse_timestamp(value: object) -> tuple[float, str]:
    if value is None:
        return 0.0, ""
    if isinstance(value, (int, float)):
        return float(value), str(value)
    text = str(value).strip()
    if not text:
        return 0.0, ""
    try:
        return float(text), text
    except ValueError:
        pass
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp(), text
    except ValueError:
        return 0.0, text


def benchmark_id_from(path: Path, result: dict) -> str | None:
    parts = path.parts
    try:
        collection = parts[parts.index("data") + 1]
    except (ValueError, IndexError):
        collection = None
    evaluation_name = clean_id(result.get("evaluation_name"))
    metric_config = result.get("metric_config") or {}
    metric_id = clean_id(metric_config.get("metric_id"))
    pieces = [part for part in [collection, evaluation_name, metric_id] if part]
    if not pieces:
        return None
    deduped = []
    for piece in pieces:
        if piece not in deduped:
            deduped.append(piece)
    return "/".join(deduped)


def metric_from(result: dict, scores_by_benchmark: dict[str, list[float]], benchmark_id: str,
                score: float) -> tuple[float, dict] | None:
    config = result.get("metric_config") or {}
    metric_id = clean_id(config.get("metric_id"))
    metric_kind = clean_id(config.get("metric_kind"))
    metric_name = clean_id(config.get("metric_name"))
    metric_unit = slug(config.get("metric_unit"))
    min_score = safe_float(config.get("min_score"))
    max_score = safe_float(config.get("max_score"))
    lower_is_better = bool(config.get("lower_is_better", False))

    name_for_type = slug(metric_kind or metric_id or metric_name or metric_unit)
    bounded_unit = metric_unit in {"proportion", "probability", "fraction", "percent", "percentage"}
    pct_like_name = name_for_type in metric_pct_names or "pass" in name_for_type
    if min_score == 0 and max_score == 1 and (bounded_unit or pct_like_name):
        if score < 0 or score > 100:
            return None
        matrix_score = score * 100.0 if score <= 1 else score
        return matrix_score, {
            "type": "pct",
            "range": [0.0, 100.0],
            "higher_is_better": not lower_is_better,
        }
    if min_score == 0 and max_score == 100 and (bounded_unit or pct_like_name):
        if score < 0 or score > 100:
            return None
        return score, {
            "type": "pct",
            "range": [0.0, 100.0],
            "higher_is_better": not lower_is_better,
        }

    if "elo" in name_for_type:
        metric_type = "elo"
    elif "rating" in name_for_type:
        metric_type = "rating"
    else:
        metric_type = name_for_type or "continuous"
    score_range = [min_score, max_score] if min_score is not None and max_score is not None else None
    return score, {
        "type": metric_type,
        "range": score_range,
        "higher_is_better": not lower_is_better,
    }


def finalize_metric_ranges(metric: dict[str, dict], scores_by_benchmark: dict[str, list[float]]) -> list[str]:
    inferred = []
    for benchmark_id, scores in scores_by_benchmark.items():
        spec = metric.setdefault(benchmark_id, {"type": "continuous", "higher_is_better": True})
        lo = min(scores)
        hi = max(scores)
        if spec.get("type") == "pct":
            if 0 <= lo and hi <= 100:
                continue
            spec["type"] = "continuous"
            spec["range"] = [float(lo), float(hi)]
            inferred.append(benchmark_id)
            continue
        declared_range = spec.get("range")
        if declared_range is not None:
            declared_lo, declared_hi = declared_range
            spec["range"] = [float(min(declared_lo, lo)), float(max(declared_hi, hi))]
            continue
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1.0)
            lo -= pad
            hi += pad
        spec["range"] = [float(lo), float(hi)]
        inferred.append(benchmark_id)
    return inferred


def build_matrix(output: Path) -> dict:
    files = local_aggregate_paths(output)
    model_ids: list[str] = []
    benchmark_ids: list[str] = []
    cells: dict[tuple[str, str], tuple[float, str, float]] = {}
    metric: dict[str, dict] = {}
    scores_by_benchmark: dict[str, list[float]] = defaultdict(list)
    skipped = Counter()
    duplicate_cells = 0

    for path in files:
        try:
            record = json.loads(path.read_text())
        except Exception:
            skipped["invalid_json"] += 1
            continue
        model_info = record.get("model_info") or {}
        model_id = clean_id(model_info.get("id") or model_info.get("name"))
        if model_id is None:
            skipped["missing_model_id"] += 1
            continue
        if model_id not in model_ids:
            model_ids.append(model_id)
        timestamp, timestamp_raw = parse_timestamp(record.get("retrieved_timestamp"))
        for result in record.get("evaluation_results") or []:
            if not isinstance(result, dict):
                skipped["non_object_result"] += 1
                continue
            benchmark_id = benchmark_id_from(path, result)
            raw_score = safe_float((result.get("score_details") or {}).get("score"))
            if benchmark_id is None or raw_score is None:
                skipped["missing_benchmark_or_score"] += 1
                continue
            converted = metric_from(result, scores_by_benchmark, benchmark_id, raw_score)
            if converted is None:
                skipped["invalid_pct_score"] += 1
                continue
            score, spec = converted
            if benchmark_id not in benchmark_ids:
                benchmark_ids.append(benchmark_id)
            scores_by_benchmark[benchmark_id].append(score)
            if benchmark_id not in metric or metric[benchmark_id].get("range") is None:
                metric[benchmark_id] = spec
            key = (model_id, benchmark_id)
            if key in cells:
                duplicate_cells += 1
            if key not in cells or timestamp >= cells[key][0]:
                cells[key] = (timestamp, timestamp_raw, score)

    if not cells:
        raise RuntimeError("No EEE aggregate scores found under raw/eee/data")

    inferred_ranges = finalize_metric_ranges(metric, scores_by_benchmark)

    scores_path = output / "scores.csv"
    with scores_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *benchmark_ids])
        for model_id in model_ids:
            row = [model_id]
            for benchmark_id in benchmark_ids:
                cell = cells.get((model_id, benchmark_id))
                row.append("" if cell is None else f"{cell[2]:.17g}")
            writer.writerow(row)

    meta_path = output / "scores.meta.json"
    meta_path.write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n")

    manifest = {
        "source": repo_id,
        "raw_dir": str(output / "raw" / "eee"),
        "aggregate_json_files": len(files),
        "models": len(model_ids),
        "benchmarks": len(benchmark_ids),
        "observed_cells": len(cells),
        "duplicate_cells_kept_latest": duplicate_cells,
        "inferred_range_benchmarks": inferred_ranges,
        "skipped": dict(skipped),
        "outputs": {
            "scores_csv": str(scores_path),
            "scores_meta_json": str(meta_path),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_inventory(output: Path, paths: list[str]) -> None:
    counts = Counter(path.split("/")[1] for path in paths)
    inventory = {
        "source": repo_id,
        "aggregate_json_files": len(paths),
        "benchmarks": len(counts),
        "benchmark_counts": dict(sorted(counts.items())),
        "paths": paths,
    }
    (output / "eee_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("~/Downloads/eee_matrix").expanduser())
    parser.add_argument("--benchmarks", nargs="*", default=None, help="EEE benchmark folders to download, e.g. gpqa-diamond hfopenllm_v2")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of aggregate JSON files to download after filtering")
    parser.add_argument("--force", action="store_true", help="Re-copy files even when raw files already exist")
    parser.add_argument("--inventory-only", action="store_true", help="Write eee_inventory.json without downloading gated files")
    parser.add_argument("--skip-download", action="store_true", help="Build scores.csv from existing raw/eee files")
    parser.add_argument("--workers", type=int, default=16, help="Parallel EEE aggregate downloads")
    args = parser.parse_args()

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    benchmarks = set(args.benchmarks) if args.benchmarks else None
    paths = eee_aggregate_paths(api, benchmarks)
    if args.limit is not None:
        paths = paths[:args.limit]
    write_inventory(output, paths)
    print(f"EEE inventory: {len(paths)} aggregate JSON files -> {output / 'eee_inventory.json'}")
    if args.inventory_only:
        return

    if not args.skip_download:
        try:
            failures = download_aggregates(paths, output, args.force, args.workers)
            if failures:
                failure_path = output / "download_failures.json"
                failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
                raise SystemExit(f"EEE download left {len(failures)} failed files; see {failure_path}")
        except (GatedRepoError, HfHubHTTPError) as exc:
            raise SystemExit(
                "Could not download EEE files. Visit https://huggingface.co/datasets/evaleval/EEE_datastore, "
                "accept access, then run `huggingface-cli login` in your terminal. "
                f"Original error: {exc.__class__.__name__}: {exc}"
            )
        print(f"EEE raw files available under {output / 'raw' / 'eee'}")

    manifest = build_matrix(output)
    print(json.dumps({
        "scores_csv": manifest["outputs"]["scores_csv"],
        "scores_meta_json": manifest["outputs"]["scores_meta_json"],
        "models": manifest["models"],
        "benchmarks": manifest["benchmarks"],
        "observed_cells": manifest["observed_cells"],
    }, indent=2))


if __name__ == "__main__":
    main()