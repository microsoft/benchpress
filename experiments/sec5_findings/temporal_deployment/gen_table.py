#!/usr/bin/env python
"""Generate the Section 5.3 temporal-deployment table from results.json."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.artifact_utils import ensure_artifacts
from benchpress.io_utils import load_json

RESULTS_PATH = os.path.join(HERE, "results.json")
TABLE_PATH = os.path.join(HERE, "table.tex")
DISPLAY_K = [1, 5, 10]
EXPECTED_PROTOCOL = "temporal_deployment_hard_rule_v4"

def fmt(value, suffix=""):
    if value is None:
        return "--"
    return f"{float(value):.1f}{suffix}"


def _ordered_landmarks(payload: dict) -> list:
    return sorted(
        payload["landmarks"],
        key=lambda row: (row["cutoff_date"], row["family_name"]),
    )


def build_table(payload: dict) -> str:
    rows = []
    landmarks = _ordered_landmarks(payload)
    for landmark in landmarks:
        key = landmark["family_key"]
        summary = payload["summary_by_family"][key]
        display_name = summary["family_name"]
        observed_counts = summary.get("target_observed_counts", {})
        observed = max(observed_counts.values()) if observed_counts else "--"
        cells = []
        for k in DISPLAY_K:
            by_k = summary["by_k"][str(k)]
            cells.append(fmt(by_k["medape"]["median"]))
            cells.append(fmt(by_k["medae"]["median"]))
        rows.append(
            f"{display_name:30s} & {observed} & {summary['n_train_models']:2d} & "
            + " & ".join(cells)
            + r" \\"
        )

    med_rows = []
    for k in DISPLAY_K:
        medape_vals = [
            payload["summary_by_family"][landmark["family_key"]]["by_k"][str(k)]["medape"]["median"]
            for landmark in landmarks
        ]
        medae_vals = [
            payload["summary_by_family"][landmark["family_key"]]["by_k"][str(k)]["medae"]["median"]
            for landmark in landmarks
        ]
        med_rows.append(fmt(_median(medape_vals)))
        med_rows.append(fmt(_median(medae_vals)))

    return "\n".join([
        r"\begin{tabular}{@{}l r r rr rr rr@{}}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{$k = 1$} & \multicolumn{2}{c}{$k = 5$} & \multicolumn{2}{c}{$k = 10$} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"Target model & Obs. & Train & MedAPE & MedAE & MedAPE & MedAE & MedAPE & MedAE \\",
        r"\midrule",
        *rows,
        r"\midrule",
        r"\textit{Median}          &    &    & " + " & ".join(med_rows) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def _median(values):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def main():
    ensure_artifacts(
        [RESULTS_PATH],
        ["{python}", os.path.join(HERE, "run.py"), "--mode", "merge"],
        description="Section 5.3 temporal-deployment results",
    )
    payload = load_json(RESULTS_PATH)
    protocol = payload.get("config", {}).get("protocol_version")
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f"{RESULTS_PATH} has protocol {protocol!r}; expected {EXPECTED_PROTOCOL!r}. "
            "Run `python run.py --mode run-all` and `python run.py --mode merge` first."
        )
    table = build_table(payload)
    with open(TABLE_PATH, "w", encoding="utf-8") as f:
        f.write(table)
    print(table)
    print(f"Wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
