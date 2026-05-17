"""Download the public BenchPress score-matrix artifacts from Hugging Face.

The release repository intentionally does not track generated
``benchpress/data/`` artifacts. This module restores the canonical JSON and
supporting evidence files used by the package from the public Hugging Face
dataset release.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


HF_REPO = "yzeng58/benchpress-score-matrix"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/raw/main"
PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"

FILES = {
    "matrix_json": "data/llm_benchmark_data.json",
    "cost_evidence": "data/benchmark_cost_evidence.json",
    "schema": "data/SCHEMA.md",
    "readme": "data/README.md",
    "cost_readme": "data/benchmark_cost_evidence.README.md",
    "metadata": "metadata.json",
}

CSV_FALLBACK_FILES = {
    "models": "data/models.csv",
    "benchmarks": "data/benchmarks.csv",
    "scores": "data/scores_all.csv",
}


def _download(url: str, path: Path, *, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    urlretrieve(url, path)


def _download_optional(url: str, path: Path, *, force: bool = False) -> bool:
    try:
        _download(url, path, force=force)
        return True
    except (HTTPError, URLError):
        return False


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _none_if_empty(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"nan", "none", "null"}:
            return None
        return stripped
    return value


def _to_float(value):
    value = _none_if_empty(value)
    if value is None:
        return None
    return float(value)


def _to_bool(value):
    value = _none_if_empty(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def _json_field(value):
    value = _none_if_empty(value)
    if value is None:
        return {}
    return json.loads(value)


def _model(row: dict) -> dict:
    return {
        "id": row["model_id"],
        "name": row["model_name"],
        "provider": row["provider"],
        "release_date": _none_if_empty(row.get("release_date")),
        "params_total_M": _to_float(row.get("params_total_M")),
        "params_active_M": _to_float(row.get("params_active_M")),
        "architecture": _none_if_empty(row.get("architecture")),
        "is_reasoning": _to_bool(row.get("is_reasoning")),
        "open_weights": _to_bool(row.get("open_weights")),
        "canonical_setting": _json_field(row.get("canonical_setting_json")),
    }


def _benchmark(row: dict) -> dict:
    return {
        "id": row["benchmark_id"],
        "name": row["benchmark_name"],
        "category": row["category"],
        "metric": _none_if_empty(row.get("metric")),
        "num_problems": _to_float(row.get("num_problems")),
        "source_url": _none_if_empty(row.get("source_url")),
        "canonical_setting": _json_field(row.get("canonical_setting_json")),
    }


def _score(row: dict) -> dict:
    return {
        "model_id": row["model_id"],
        "benchmark_id": row["benchmark_id"],
        "score": _to_float(row.get("score")),
        "reference_url": _none_if_empty(row.get("reference_url")),
        "reported_setting": _json_field(row.get("reported_setting_json")),
        "matches_canonical": _to_bool(row.get("matches_canonical")),
        "source_type": _none_if_empty(row.get("source_type")),
        "audit_status": _none_if_empty(row.get("audit_status")),
        "notes": _none_if_empty(row.get("notes")) or "",
    }


def _build_json_from_csv(cache_dir: Path, output_path: Path) -> None:
    data = {
        "models": [_model(row) for row in _read_csv(cache_dir / CSV_FALLBACK_FILES["models"])],
        "benchmarks": [_benchmark(row) for row in _read_csv(cache_dir / CSV_FALLBACK_FILES["benchmarks"])],
        "scores": [_score(row) for row in _read_csv(cache_dir / CSV_FALLBACK_FILES["scores"])],
        "generated": {
            "source": f"https://huggingface.co/datasets/{HF_REPO}",
            "note": "Rebuilt from the public CSV mirror; rich candidates/cost evidence require the canonical JSON artifact.",
        },
    }
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def download_data(data_dir: Path = PACKAGE_DATA_DIR, *, force: bool = False) -> Path:
    """Download public data artifacts into ``benchpress/data/``."""
    matrix_path = data_dir / "llm_benchmark_data.json"
    exact_json_available = _download_optional(
        f"{HF_BASE_URL}/{FILES['matrix_json']}", matrix_path, force=force)
    if exact_json_available:
        for key, rel_path in FILES.items():
            if key == "matrix_json":
                continue
            _download_optional(f"{HF_BASE_URL}/{rel_path}", data_dir / Path(rel_path).name, force=force)
        return matrix_path

    cache_dir = data_dir / "_hf_cache"
    for rel_path in CSV_FALLBACK_FILES.values():
        _download(f"{HF_BASE_URL}/{rel_path}", cache_dir / rel_path, force=force)
    _build_json_from_csv(cache_dir, matrix_path)
    return matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BenchPress score-matrix data from Hugging Face.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PACKAGE_DATA_DIR,
        help="Directory where benchpress data files should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download CSV files even if cached copies already exist.",
    )
    args = parser.parse_args()
    path = download_data(args.data_dir, force=args.force)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
