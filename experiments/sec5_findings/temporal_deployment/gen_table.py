#!/usr/bin/env python
"""Generate the Appendix C temporal-deployment table from results.json."""

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

# Display order: Claude -> GPT -> DeepSeek -> Gemini -> Qwen.
# Within each provider, keep chronological order (smaller train count first).
DISPLAY_ORDER = [
    "claude_sonnet_opus_4",
    "claude_sonnet_4_5",
    "gpt_4_1_family",
    "gpt_5",
    "gpt_5_1",
    "deepseek_r1",
    "gemini_2_5_pro",
    "qwen_3",
]

# Override family display names so multi-variant landmarks are explicitly marked
# as a "family" (the row pools all listed target variants).
DISPLAY_NAME_OVERRIDE: dict[str, str] = {
    "gpt_4_1_family": "GPT-4.1",
}


def fmt(value, suffix=""):
    if value is None:
        return "--"
    return f"{float(value):.1f}{suffix}"


def _ordered_landmarks(payload: dict) -> list:
    by_key = {l["family_key"]: l for l in payload["landmarks"]}
    ordered = [by_key[k] for k in DISPLAY_ORDER if k in by_key]
    # Append any landmarks not in DISPLAY_ORDER, preserving original order.
    seen = set(DISPLAY_ORDER)
    ordered.extend(l for l in payload["landmarks"] if l["family_key"] not in seen)
    return ordered


def build_table(payload: dict) -> str:
    rows = []
    landmarks = _ordered_landmarks(payload)
    for landmark in landmarks:
        key = landmark["family_key"]
        summary = payload["summary_by_family"][key]
        display_name = DISPLAY_NAME_OVERRIDE.get(key, summary["family_name"])
        cells = []
        for k in DISPLAY_K:
            by_k = summary["by_k"][str(k)]
            cells.append(fmt(by_k["medape"]["median"]))
            cells.append(fmt(by_k["medae"]["median"]))
        rows.append(
            f"{display_name:30s} & {summary['n_target_models']:2d} & {summary['n_train_models']:2d} & "
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
        r"Landmark & \# Variants & Train & MedAPE & MedAE & MedAPE & MedAE & MedAPE & MedAE \\",
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
        description="Section 5.4 temporal-deployment results",
    )
    payload = load_json(RESULTS_PATH)
    table = build_table(payload)
    with open(TABLE_PATH, "w", encoding="utf-8") as f:
        f.write(table)
    print(table)
    print(f"Wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
