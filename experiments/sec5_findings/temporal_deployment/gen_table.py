#!/usr/bin/env python
"""Generate the Section 5.3 temporal-deployment table from results.json."""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.io_utils import load_json, write_json
from summary_utils import (
    ordered_landmarks,
    temporal_family_k_summary,
    temporal_overall_medians,
)

RESULTS_PATH = os.path.join(HERE, "results.json")
TABLE_PATH = os.path.join(HERE, "table.tex")
DISPLAY_K = [1, 5, 10]
EXPECTED_PROTOCOL = "temporal_deployment_hard_rule_v4"

def fmt(value, suffix=""):
    if value is None:
        return "--"
    return f"{float(value):.1f}{suffix}"


def build_table(payload: dict, hidden_only: bool = False) -> str:
    rows = []
    landmarks = ordered_landmarks(payload)
    hidden_summary = temporal_family_k_summary(payload, hidden_only=True) if hidden_only else None
    for landmark in landmarks:
        key = landmark["family_key"]
        summary = hidden_summary[key] if hidden_only else payload["summary_by_family"][key]
        display_name = summary["family_name"]
        observed_counts = summary.get("target_observed_counts", {})
        observed = (
            summary.get("observed")
            if hidden_only
            else max(observed_counts.values()) if observed_counts else "--"
        )
        cells = []
        for k in DISPLAY_K:
            by_k = summary["by_k"][str(k)]
            if hidden_only:
                cells.append(fmt(by_k["medape"]))
                cells.append(fmt(by_k["medae"]))
            else:
                cells.append(fmt(by_k["medape"]["median"]))
                cells.append(fmt(by_k["medae"]["median"]))
        rows.append(
            f"{display_name:30s} & {observed} & {summary['n_train_models']:2d} & "
            + " & ".join(cells)
            + r" \\"
        )

    med_rows = []
    if hidden_only:
        medians = temporal_overall_medians(payload, hidden_only=True)
        for k in DISPLAY_K:
            med_rows.append(fmt(medians["medape"][k]))
            med_rows.append(fmt(medians["medae"][k]))
    else:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-only", action="store_true",
                        help="Build the appendix table from non-revealed finite predictions only.")
    parser.add_argument("--out", default=None,
                        help="Output path for the LaTeX tabular body.")
    parser.add_argument(
        "--hidden-summary-json",
        default=None,
        help="Optional JSON path for hidden-only summaries already computed by run.py --mode merge.",
    )
    args = parser.parse_args()
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing merged results JSON: {RESULTS_PATH}")
    payload = load_json(RESULTS_PATH)
    protocol = payload.get("config", {}).get("protocol_version")
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f"{RESULTS_PATH} has protocol {protocol!r}; expected {EXPECTED_PROTOCOL!r}. "
            "Run `python run.py --mode run-all` and `python run.py --mode merge` first."
        )
    table = build_table(payload, hidden_only=args.hidden_only)
    out_path = args.out or (
        os.path.join(HERE, "table_hidden_only.tex") if args.hidden_only else TABLE_PATH
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(table)
    print(f"Wrote {out_path}")
    if args.hidden_summary_json:
        write_json(
            args.hidden_summary_json,
            {
                "summary_hidden_only_by_k": temporal_overall_medians(payload, hidden_only=True),
                "summary_hidden_only_by_family": temporal_family_k_summary(payload, hidden_only=True),
            },
            indent=2,
            trailing_newline=True,
        )
        print(f"Wrote {args.hidden_summary_json}")


if __name__ == "__main__":
    main()
