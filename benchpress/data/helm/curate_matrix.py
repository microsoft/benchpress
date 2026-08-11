#!/usr/bin/env python
"""Curate HELM group tables into a BenchPress-compatible score matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


release_configs = {
    "lite": {
        "name": "HELM Lite v1.13.0",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.13.0",
    },
    "classic": {
        "name": "HELM Classic v0.4.0",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/classic/benchmark_output/releases/v0.4.0",
    },
    "capabilities": {
        "name": "HELM Capabilities v1.15.0",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/releases/v1.15.0",
    },
    "long_context": {
        "name": "HELM Long Context v1.0.0",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/long-context/benchmark_output/releases/v1.0.0",
    },
}

default_releases = ["lite", "classic", "capabilities", "long_context"]
skip_table_titles = {"general information", "annotation"}
model_header_names = {"model", "model/adapter"}
lower_better_terms = {
    "calibration",
    "ece",
    "efficiency",
    "inference time",
    "latency",
    "toxic",
    "toxicity",
    "bias",
    "stereotype",
    "representation",
}
latency_terms = {"inference time", "latency", "time (s)"}
metadata_column_terms = {"# eval", "# train", "# prompt tokens", "# output tokens", "# trials", "# instances", "# references", "truncated"}


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return re.sub(r"\s+", " ", text) or None


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "nan", "none", "null", "infinity", "+infinity", "-infinity"}:
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def benchmark_component(value: object) -> str:
    text = clean_text(value) or "unknown"
    return text.replace("/", "_")


def split_dataset_metric(column_name: str) -> tuple[str, str]:
    if " - " not in column_name:
        return "Overall", column_name
    dataset_name, metric_name = column_name.split(" - ", 1)
    return dataset_name, metric_name


def download_json(url: str, path: str, force: bool) -> object:
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    with open(path, "wb") as f:
        f.write(content)
    return json.loads(content)


def group_ids_from_index(groups_json: object, include_drilldowns: bool) -> list[str]:
    group_ids = []
    if not isinstance(groups_json, list):
        return group_ids
    for table in groups_json:
        if not include_drilldowns and table.get("title") != "All scenarios":
            continue
        if not isinstance(table, dict):
            continue
        for row in table.get("rows") or []:
            if not row:
                continue
            href = row[0].get("href") if isinstance(row[0], dict) else None
            match = re.search(r"(?:\?|&)group=([^&]+)", href or "")
            if match and match.group(1) not in group_ids:
                group_ids.append(match.group(1))
    return group_ids


def release_group_ids(release: str, output: str, force: bool, include_drilldowns: bool) -> list[str]:
    config = release_configs[release]
    release_dir = os.path.join(output, "raw", "helm", release)
    groups_index_path = os.path.join(release_dir, "groups.json")
    groups_json = download_json(f"{config['base_url']}/groups.json", groups_index_path, force)
    return group_ids_from_index(groups_json, include_drilldowns)


def download_release(release: str, output: str, force: bool, workers: int,
                     include_drilldowns: bool) -> tuple[list[str], list[tuple[str, str]]]:
    config = release_configs[release]
    release_dir = os.path.join(output, "raw", "helm", release)
    group_ids = release_group_ids(release, output, force, include_drilldowns)
    failures = []

    def download_group(group_id: str) -> tuple[str, str] | None:
        url = f"{config['base_url']}/groups/{group_id}.json"
        path = os.path.join(release_dir, "groups", f"{group_id}.json")
        try:
            download_json(url, path, force)
            return None
        except Exception as exc:
            return group_id, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_group, group_id) for group_id in group_ids]
        for future in as_completed(futures):
            failure = future.result()
            if failure is not None:
                failures.append(failure)
    return group_ids, failures


def selected_group_ids(output: str, releases: list[str], include_drilldowns: bool) -> dict[str, list[str]]:
    return {
        release: release_group_ids(release, output, False, include_drilldowns)
        for release in releases
    }


def local_group_files(output: str, groups_by_release: dict[str, list[str]]) -> list[tuple[str, str]]:
    files = []
    for release, group_ids in groups_by_release.items():
        groups_dir = os.path.join(output, "raw", "helm", release, "groups")
        if not os.path.isdir(groups_dir):
            continue
        for group_id in group_ids:
            path = os.path.join(groups_dir, f"{group_id}.json")
            if os.path.exists(path):
                files.append((release, path))
    return files


def column_is_lower_better(table_title: str, column_name: str) -> bool:
    text = f"{table_title} {column_name}".lower()
    return any(term in text for term in lower_better_terms)


def column_is_latency(table_title: str, column_name: str) -> bool:
    text = f"{table_title} {column_name}".lower()
    return any(term in text for term in latency_terms)


def column_is_metadata(column_name: str) -> bool:
    text = column_name.lower()
    return any(term in text for term in metadata_column_terms)


def score_range(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1.0)
        lo -= pad
        hi += pad
    return [float(lo), float(hi)]


def metric_for_column(table_title: str, column_name: str, values: list[float]) -> tuple[dict, float]:
    lo = min(values)
    hi = max(values)
    higher_is_better = not column_is_lower_better(table_title, column_name)
    if column_is_latency(table_title, column_name):
        return {
            "type": "latency",
            "range": [float(min(0.0, lo)), float(hi)],
            "higher_is_better": False,
        }, 1.0
    if lo >= 0 and hi <= 1:
        return {
            "type": "pct",
            "range": [0.0, 100.0],
            "higher_is_better": higher_is_better,
        }, 100.0
    if lo >= 0 and hi <= 100:
        return {
            "type": "pct",
            "range": [0.0, 100.0],
            "higher_is_better": higher_is_better,
        }, 1.0
    return {
        "type": "continuous",
        "range": score_range(values),
        "higher_is_better": higher_is_better,
    }, 1.0


def extract_tables(output: str, groups_by_release: dict[str, list[str]]) -> tuple[dict, list[str], list[str], dict, Counter, int]:
    cells = {}
    model_ids = []
    benchmark_ids = []
    columns = {}
    skipped = Counter()
    duplicate_cells = 0
    for release, path in local_group_files(output, groups_by_release):
        group_id = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as f:
                tables = json.load(f)
        except Exception:
            skipped["invalid_json"] += 1
            continue
        if not isinstance(tables, list):
            skipped["non_list_group_file"] += 1
            continue
        for table in tables:
            if not isinstance(table, dict):
                skipped["non_object_table"] += 1
                continue
            table_title = clean_text(table.get("title")) or "scores"
            if table_title.lower() in skip_table_titles:
                skipped["metadata_table"] += 1
                continue
            headers = [clean_text(cell.get("value")) if isinstance(cell, dict) else None for cell in table.get("header") or []]
            if not headers or (headers[0] or "").lower() not in model_header_names:
                skipped["non_model_table"] += 1
                continue
            for row in table.get("rows") or []:
                if not row:
                    continue
                first_cell = row[0] if isinstance(row[0], dict) else {}
                model_id = clean_text(first_cell.get("value"))
                if model_id is None:
                    skipped["missing_model_id"] += 1
                    continue
                for index, column_name in enumerate(headers[1:], start=1):
                    if column_name is None:
                        continue
                    if column_is_metadata(column_name):
                        skipped["metadata_column"] += 1
                        continue
                    cell = row[index] if index < len(row) and isinstance(row[index], dict) else {}
                    value = safe_float(cell.get("value"))
                    if value is None:
                        skipped["missing_score"] += 1
                        continue
                    dataset_name, metric_name = split_dataset_metric(column_name)
                    benchmark_id = "/".join([
                        benchmark_component(f"{dataset_name} ({release}: {group_id})"),
                        benchmark_component(table_title),
                        benchmark_component(metric_name),
                    ])
                    if benchmark_id not in benchmark_ids:
                        benchmark_ids.append(benchmark_id)
                    columns[benchmark_id] = (table_title, column_name)
                    if model_id not in model_ids:
                        model_ids.append(model_id)
                    key = (model_id, benchmark_id)
                    if key in cells:
                        duplicate_cells += 1
                    cells[key] = value
    return cells, model_ids, benchmark_ids, columns, skipped, duplicate_cells


def write_matrix(output: str, releases: list[str], include_drilldowns: bool) -> dict:
    groups_by_release = selected_group_ids(output, releases, include_drilldowns)
    raw_cells, model_ids, benchmark_ids, columns, skipped, duplicate_cells = extract_tables(output, groups_by_release)
    if not raw_cells:
        raise RuntimeError("No HELM scores found under raw/helm")
    values_by_benchmark = defaultdict(list)
    for (_, benchmark_id), score in raw_cells.items():
        values_by_benchmark[benchmark_id].append(score)

    metric = {}
    scale_by_benchmark = {}
    for benchmark_id in benchmark_ids:
        table_title, column_name = columns[benchmark_id]
        spec, scale = metric_for_column(table_title, column_name, values_by_benchmark[benchmark_id])
        metric[benchmark_id] = spec
        scale_by_benchmark[benchmark_id] = scale

    scores_path = os.path.join(output, "scores.csv")
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *benchmark_ids])
        for model_id in model_ids:
            row = [model_id]
            for benchmark_id in benchmark_ids:
                value = raw_cells.get((model_id, benchmark_id))
                row.append("" if value is None else f"{value * scale_by_benchmark[benchmark_id]:.17g}")
            writer.writerow(row)

    meta_path = os.path.join(output, "scores.meta.json")
    with open(meta_path, "w") as f:
        json.dump(metric, f, indent=2, sort_keys=True)
        f.write("\n")

    manifest = {
        "source": "HELM public benchmark_output releases",
        "releases": {release: release_configs[release] for release in releases},
        "groups": groups_by_release,
        "include_drilldowns": include_drilldowns,
        "raw_dir": os.path.join(output, "raw", "helm"),
        "group_json_files": len(local_group_files(output, groups_by_release)),
        "models": len(model_ids),
        "benchmarks": len(benchmark_ids),
        "observed_cells": len(raw_cells),
        "duplicate_cells_kept_latest": duplicate_cells,
        "skipped": dict(skipped),
        "outputs": {
            "scores_csv": scores_path,
            "scores_meta_json": meta_path,
        },
    }
    with open(os.path.join(output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def write_inventory(output: str, releases: list[str], groups_by_release: dict[str, list[str]],
                    include_drilldowns: bool) -> None:
    inventory = {
        "source": "HELM public benchmark_output releases",
        "releases": {release: release_configs[release] for release in releases},
        "groups": groups_by_release,
        "include_drilldowns": include_drilldowns,
    }
    with open(os.path.join(output, "helm_inventory.json"), "w") as f:
        json.dump(inventory, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.expanduser("~/Downloads/helm_matrix"))
    parser.add_argument("--releases", nargs="*", choices=sorted(release_configs), default=default_releases)
    parser.add_argument("--force", action="store_true", help="Redownload raw HELM JSON files even when present")
    parser.add_argument("--inventory-only", action="store_true", help="Download groups.json files and write inventory only")
    parser.add_argument("--skip-download", action="store_true", help="Build scores.csv from existing raw/helm files")
    parser.add_argument("--include-drilldowns", action="store_true", help="Also include scenario, category, and ablation drilldown groups")
    parser.add_argument("--workers", type=int, default=8, help="Parallel HELM group-table downloads")
    args = parser.parse_args()

    output = os.path.expanduser(args.output)
    os.makedirs(output, exist_ok=True)
    groups_by_release = {}
    failures = {}
    if not args.skip_download:
        for release in args.releases:
            group_ids, release_failures = download_release(
                release, output, args.force, args.workers, args.include_drilldowns)
            groups_by_release[release] = group_ids
            if release_failures:
                failures[release] = release_failures
    else:
        groups_by_release = selected_group_ids(output, args.releases, args.include_drilldowns)

    write_inventory(output, args.releases, groups_by_release, args.include_drilldowns)
    if not args.skip_download and failures:
            failure_path = os.path.join(output, "download_failures.json")
            with open(failure_path, "w") as f:
                json.dump(failures, f, indent=2, sort_keys=True)
                f.write("\n")
            raise SystemExit(f"HELM download left failed group files; see {failure_path}")

    print(f"HELM raw files available under {os.path.join(output, 'raw', 'helm')}")
    if args.inventory_only:
        return

    manifest = write_matrix(output, args.releases, args.include_drilldowns)
    print(json.dumps({
        "scores_csv": manifest["outputs"]["scores_csv"],
        "scores_meta_json": manifest["outputs"]["scores_meta_json"],
        "models": manifest["models"],
        "benchmarks": manifest["benchmarks"],
        "observed_cells": manifest["observed_cells"],
    }, indent=2))


if __name__ == "__main__":
    main()