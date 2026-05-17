#!/usr/bin/env python3
"""Generate the §4.3 model-side error-analysis table from per-H results.

H1–H4: univariate Spearman ρ (correlational)
H5–H8: paired Wilcoxon Δ and p-value (ablation)
H9: temporal comparison summary
"""
import os
from benchpress.io_utils import load_json

HERE = os.path.dirname(os.path.abspath(__file__))

CORRELATIONAL = [
    ("H1", "H1_model_size",          "log_params"),
    ("H2", "H2_model_type",           "is_reasoning"),
    ("H3", "H3_score_level",         "med_true"),
    ("H4", "H4_rank2_expressibility","rank2_R2"),
]

ABLATION = [
    ("H5", "H5_neighbor_quality"),
    ("H6", "H6_neighbor_evidence"),
    ("H7", "H7_family_peers"),
]

TARGETS = ["medape", "medae"]
ABL_METRICS = ["medape", "medae"]


def main():
    print("=== Correlational (H1–H4): Spearman ρ ===")
    print(f"{'H':5s}" + "".join(f"  {t:>20s}" for t in TARGETS))
    print("-" * (5 + 22 * len(TARGETS)))
    for label, dirname, feature in CORRELATIONAL:
        path = os.path.join(HERE, dirname, "results.json")
        if not os.path.exists(path):
            print(f"{label:5s}  [no results]")
            continue
        data = load_json(path)
        cells = []
        for t in TARGETS:
            s = data.get("univariate_spearman", {}).get(t, {}).get(feature, {})
            rho = s.get("rho", float("nan"))
            p = s.get("p", float("nan"))
            cells.append(f"ρ={rho:+.3f} p={p:.1e}")
        print(f"{label:5s}" + "".join(f"  {c:>20s}" for c in cells))

    print(f"\n=== Ablation (H5–H7): Paired Wilcoxon Δ ===")
    print(f"{'H':5s}" + "".join(f"  {m:>22s}" for m in ABL_METRICS))
    print("-" * (5 + 24 * len(ABL_METRICS)))
    for label, dirname in ABLATION:
        path = os.path.join(HERE, dirname, "results.json")
        if not os.path.exists(path):
            print(f"{label:5s}  [no results]")
            continue
        data = load_json(path)
        tests = data.get("tests", {})
        cells = []
        for m in ABL_METRICS:
            s = tests.get(m, {})
            md = s.get("median_delta", float("nan"))
            p = s.get("p", float("nan"))
            cells.append(f"Δ={md:+.2f} p={p:.3f}")
        print(f"{label:5s}" + "".join(f"  {c:>22s}" for c in cells))

    # H8 (two conditions)
    path_h8 = os.path.join(HERE, "H8_observation_count", "results.json")
    if os.path.exists(path_h8):
        h8 = load_json(path_h8)
        for cond, key in [("H8 25%", "hide_25pct"), ("H8 75%", "hide_75pct")]:
            tests = h8.get(key, {}).get("tests", {})
            cells = []
            for m in ABL_METRICS:
                s = tests.get(m, {})
                md = s.get("median_delta", float("nan"))
                p = s.get("p", float("nan"))
                cells.append(f"Δ={md:+.2f} p={p:.3f}")
            print(f"{cond:5s}" + "".join(f"  {c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
