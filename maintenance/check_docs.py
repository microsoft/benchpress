#!/usr/bin/env python
"""Check public README matrix counts against the current BenchPress data cache."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "benchpress" / "data" / "llm_benchmark_data.json"
DEFAULT_README = ROOT / "README.md"


@dataclass(frozen=True)
class Check:
    name: str
    expected: Any
    actual: Any
    location: str

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source JSON: {rel(path)}. Run `python -m benchpress.download_data` first."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract(pattern: str, text: str, location: str) -> tuple[str, ...]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find README claim for {location}")
    return match.groups()


def to_int(value: str) -> int:
    return int(value.replace(",", ""))


def actual_counts(data: dict[str, Any], json_path: Path) -> dict[str, Any]:
    os.environ["BENCHPRESS_DATA"] = str(json_path)
    from benchpress.build_benchmark_matrix import load_score_matrix

    _, info = load_score_matrix(json_path=json_path, return_info=True)
    providers = {m.get("provider") for m in data.get("models", []) if m.get("provider")}
    categories = {b.get("category") for b in data.get("benchmarks", []) if b.get("category")}
    return {
        "raw_models": len(data.get("models", [])),
        "providers": len(providers),
        "raw_benchmarks": len(data.get("benchmarks", [])),
        "categories": len(categories),
        "raw_scores": len(data.get("scores", [])),
        "paper_models": info.n_models,
        "paper_benchmarks": info.n_benchmarks,
        "paper_observations": info.n_observations,
        "paper_fill_rate_pct": round(info.fill_rate * 100, 1),
    }


def readme_claims(readme_text: str) -> dict[str, Any]:
    raw_models, providers = extract(
        r"\*\*([\d,]+)\s+frontier LLMs\*\*\s+from\s+([\d,]+)\s+providers",
        readme_text,
        "raw model/provider count",
    )
    raw_benchmarks, categories = extract(
        r"\*\*([\d,]+)\s+benchmarks\*\*\s+across\s+([\d,]+)\s+categories",
        readme_text,
        "raw benchmark/category count",
    )
    (raw_scores,) = extract(
        r"\*\*([\d,]+)\s+observed scores\*\*",
        readme_text,
        "raw score count",
    )
    paper_models, paper_benchmarks, paper_observations, fill_rate = extract(
        r"\*\*([\d,]+)\s+models\s*[×x]\s*([\d,]+)\s+benchmarks\*\*,\s+([\d,]+)\s+observed\s+\(([\d.]+)%\s+fill rate\)",
        readme_text,
        "paper matrix count",
    )
    alt_models, alt_benchmarks, alt_fill_rate = extract(
        r"score matrix observation pattern \(([\d,]+)\s+models\s*[×x]\s*([\d,]+)\s+benchmarks,\s+([\d.]+)%\s+filled\)",
        readme_text,
        "score-matrix image alt text",
    )
    caption_models, caption_benchmarks = extract(
        r"paper-canonical\s+([\d,]+)\s*[×x]\s*([\d,]+)\s+score matrix",
        readme_text,
        "score-matrix caption",
    )
    hf_models, hf_benchmarks = extract(
        r"`scores_paper`\s+\(the paper-canonical\s+([\d,]+)-model\s+x\s+([\d,]+)-benchmark matrix\)",
        readme_text,
        "Hugging Face scores_paper description",
    )
    return {
        "raw_models": to_int(raw_models),
        "providers": to_int(providers),
        "raw_benchmarks": to_int(raw_benchmarks),
        "categories": to_int(categories),
        "raw_scores": to_int(raw_scores),
        "paper_models": to_int(paper_models),
        "paper_benchmarks": to_int(paper_benchmarks),
        "paper_observations": to_int(paper_observations),
        "paper_fill_rate_pct": float(fill_rate),
        "alt_paper_models": to_int(alt_models),
        "alt_paper_benchmarks": to_int(alt_benchmarks),
        "alt_paper_fill_rate_pct": float(alt_fill_rate),
        "caption_paper_models": to_int(caption_models),
        "caption_paper_benchmarks": to_int(caption_benchmarks),
        "hf_paper_models": to_int(hf_models),
        "hf_paper_benchmarks": to_int(hf_benchmarks),
    }


def build_checks(actual: dict[str, Any], claims: dict[str, Any]) -> list[Check]:
    return [
        Check("raw models", actual["raw_models"], claims["raw_models"], "Step 2 bullet"),
        Check("providers", actual["providers"], claims["providers"], "Step 2 bullet"),
        Check("raw benchmarks", actual["raw_benchmarks"], claims["raw_benchmarks"], "Step 2 bullet"),
        Check("categories", actual["categories"], claims["categories"], "Step 2 bullet"),
        Check("raw score rows", actual["raw_scores"], claims["raw_scores"], "Step 2 bullet"),
        Check("paper models", actual["paper_models"], claims["paper_models"], "paper-canonical bullet"),
        Check("paper benchmarks", actual["paper_benchmarks"], claims["paper_benchmarks"], "paper-canonical bullet"),
        Check("paper observations", actual["paper_observations"], claims["paper_observations"], "paper-canonical bullet"),
        Check("paper fill rate %", actual["paper_fill_rate_pct"], claims["paper_fill_rate_pct"], "paper-canonical bullet"),
        Check("image alt paper models", actual["paper_models"], claims["alt_paper_models"], "score_matrix.png alt"),
        Check("image alt paper benchmarks", actual["paper_benchmarks"], claims["alt_paper_benchmarks"], "score_matrix.png alt"),
        Check("image alt fill rate %", actual["paper_fill_rate_pct"], claims["alt_paper_fill_rate_pct"], "score_matrix.png alt"),
        Check("caption paper models", actual["paper_models"], claims["caption_paper_models"], "score_matrix.png caption"),
        Check("caption paper benchmarks", actual["paper_benchmarks"], claims["caption_paper_benchmarks"], "score_matrix.png caption"),
        Check("HF scores_paper models", actual["paper_models"], claims["hf_paper_models"], "HF dataset description"),
        Check("HF scores_paper benchmarks", actual["paper_benchmarks"], claims["hf_paper_benchmarks"], "HF dataset description"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check README counts against BenchPress data.")
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    args = parser.parse_args()

    data = load_json(args.json_path)
    readme_text = args.readme.read_text(encoding="utf-8")
    actual = actual_counts(data, args.json_path)
    claims = readme_claims(readme_text)
    checks = build_checks(actual, claims)
    failures = [check for check in checks if not check.ok]

    print("| Check | Expected from data | README | Location | Status |")
    print("|---|---:|---:|---|---|")
    for check in checks:
        status = "ok" if check.ok else "mismatch"
        print(f"| {check.name} | {check.expected} | {check.actual} | {check.location} | {status} |")

    if failures:
        print(f"\nREADME count mismatches: {len(failures)}")
        return 1
    print("\nREADME counts match current data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
