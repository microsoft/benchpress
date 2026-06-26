#!/usr/bin/env python
"""Report stale BenchPress maintenance artifacts.

The script is read-only except for writing a Markdown report under
``maintenance/reports``. It intentionally does not run expensive experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAINTENANCE_DIR = ROOT / "maintenance"
DEFAULT_CONFIG = MAINTENANCE_DIR / "config.json"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def fmt_time(ts: float | None) -> str:
    if ts is None:
        return "missing"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def run_probe(cmd: list[str]) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode == 0, proc.stdout.strip()


def artifact_status(path: Path, sources: list[Path]) -> dict[str, Any]:
    artifact_ts = mtime(path)
    source_times = [mtime(p) for p in sources if p.exists()]
    newest_source = max(source_times) if source_times else None
    if artifact_ts is None:
        state = "missing"
    elif newest_source is not None and artifact_ts < newest_source:
        state = "stale"
    else:
        state = "ok"
    return {
        "path": rel(path),
        "state": state,
        "modified": fmt_time(artifact_ts),
        "newest_source": fmt_time(newest_source),
    }


def website_artifact_status(path: Path, raw_path: Path) -> dict[str, Any]:
    base = artifact_status(path, [])
    if not path.exists() or not raw_path.exists():
        return base

    try:
        site = load_json(path)
        from benchpress.build_benchmark_matrix import load_score_matrix

        df = load_score_matrix(json_path=raw_path)
        site_models = [m.get("id") for m in site.get("models", [])]
        site_benchmarks = [b.get("id") for b in site.get("benchmarks", [])]
        matrix_models = list(df.index)
        matrix_benchmarks = list(df.columns)
    except Exception as exc:
        base["state"] = "check_failed"
        base["newest_source"] = f"semantic check failed: {exc}"
        return base

    if set(site_models) == set(matrix_models) and set(site_benchmarks) == set(matrix_benchmarks):
        base["state"] = "ok"
        base["newest_source"] = "semantic ID match"
    else:
        base["state"] = "stale"
        base["newest_source"] = (
            f"semantic mismatch: site {len(site_models)}x{len(site_benchmarks)}, "
            f"matrix {len(matrix_models)}x{len(matrix_benchmarks)}"
        )
    return base


def matrix_summary(raw_path: Path) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    actions: list[str] = []
    if not raw_path.exists():
        actions.append(
            f"Add or download `{rel(raw_path)}` before refreshing matrix-dependent artifacts."
        )
        return lines, actions

    data = load_json(raw_path)
    scores = data.get("scores", [])
    status_counts = Counter((row.get("audit_status") or "pending") for row in scores)
    lines.extend(
        [
            f"- Raw models: {len(data.get('models', []))}",
            f"- Raw benchmarks: {len(data.get('benchmarks', []))}",
            f"- Raw score rows: {len(scores)}",
            "- Audit statuses: "
            + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())),
        ]
    )

    ok, output = run_probe([sys.executable, "-m", "benchpress.build_benchmark_matrix"])
    if ok:
        lines.append("```text\n" + output + "\n```")
    else:
        lines.append("```text\n" + output + "\n```")
        actions.append("Fix matrix build/import errors before running downstream refresh steps.")
    return lines, actions


def write_report(config: dict[str, Any], report_path: Path) -> tuple[list[str], bool]:
    raw_path = ROOT / config["raw_matrix"]
    matrix_export = ROOT / config["matrix_export"]
    website_data = ROOT / config["website_data"]
    interval_script = ROOT / config["website_interval_script"]
    default_prediction_dir = ROOT / config["default_prediction_dir"]

    actions: list[str] = []
    lines: list[str] = [
        "# BenchPress maintenance report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Repository: {ROOT}",
        "",
        "## Matrix",
    ]

    matrix_lines, matrix_actions = matrix_summary(raw_path)
    lines.extend(matrix_lines or [f"- `{rel(raw_path)}` is missing."])
    actions.extend(matrix_actions)

    tracked = [
        artifact_status(matrix_export, [raw_path]),
        website_artifact_status(website_data, raw_path),
    ]
    metadata_path = default_prediction_dir / "metadata.json"
    predictions_path = default_prediction_dir / "predictions.npz"
    tracked.append(artifact_status(metadata_path, [raw_path]))
    tracked.append(artifact_status(predictions_path, [raw_path]))

    lines.extend(["", "## Core artifacts", "", "| Artifact | State | Modified | Newest source |", "|---|---|---|---|"])
    for row in tracked:
        lines.append(
            f"| `{row['path']}` | {row['state']} | {row['modified']} | {row['newest_source']} |"
        )
        if row["state"] != "ok":
            actions.append(f"Refresh `{row['path']}` ({row['state']}).")

    lines.extend(["", "## Probe sets", "", "| Probe set | State | Output | Modified |", "|---|---|---|---|"])
    for item in config.get("probe_sets", []):
        output = ROOT / item["output"]
        status = artifact_status(output, [raw_path])
        lines.append(
            f"| {item['name']} | {status['state']} | `{status['path']}` | {status['modified']} |"
        )
        if status["state"] != "ok":
            actions.append(f"Rerun probe set `{item['name']}`.")

    lines.extend(["", "## Website", ""])
    if not interval_script.exists():
        lines.append(f"- Interval script missing: `{rel(interval_script)}`")
        actions.append("Restore or replace the website interval post-processing script.")
    else:
        lines.append(f"- Interval post-processor: `{rel(interval_script)}`")
    if website_data.exists():
        try:
            site = load_json(website_data)
            lines.extend(
                [
                    f"- Website models: {len(site.get('models', []))}",
                    f"- Website benchmarks: {len(site.get('benchmarks', []))}",
                    f"- Has prediction intervals: {'prediction_intervals' in site}",
                    f"- Has trust probabilities: {'trust_probabilities' in site}",
                ]
            )
        except json.JSONDecodeError as exc:
            lines.append(f"- Website data JSON parse failed: {exc}")
            actions.append("Fix `website/data.json` JSON syntax.")
    else:
        actions.append("Regenerate `website/data.json`.")

    lines.extend(["", "## Suggested actions"])
    if actions:
        lines.extend(f"- {action}" for action in actions)
    else:
        lines.append("- No stale or missing tracked artifacts detected.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    return actions, not actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or (MAINTENANCE_DIR / "reports" / f"update_report_{stamp}.md")
    actions, clean = write_report(config, out)
    print(f"Wrote {rel(out)}")
    print(f"Suggested actions: {len(actions)}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
