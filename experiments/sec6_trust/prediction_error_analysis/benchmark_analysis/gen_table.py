#!/usr/bin/env python3
"""Generate the §4.3 benchmark-side error-analysis table from per-H results.

Top block (H1-H3): correlational hypotheses — univariate Spearman ρ vs targets.
Bottom block (H4-H7): ablation hypotheses — paired Wilcoxon Δ vs baseline.
Random seeds are set inside the corresponding H analyze/ablation scripts; this
file only reads their stored results and formats the paper-facing summary.
"""
import os
import numpy as np
from benchpress.io_utils import load_json

HERE = os.path.dirname(os.path.abspath(__file__))

CORR_HYPOTHESES = [
    ("H1", "Low-rank fit",       "H1_low_rank_fit",         "rank2_R2"),
    ("H2", "Score level",        "H2_score_level",          "med_score"),
    ("H3", "Score spread",       "H3_score_spread",         "std_score"),
]

ABL_HYPOTHESES = [
    ("H4", "Target coverage",         "H4_target_coverage",         None),
    ("H5", "Strong-neighbor presence","H5_strong_neighbor_presence","0.85"),
    ("H6", "Strong-neighbor support", "H6_strong_neighbor_support", None),
    ("H7", "Same-category evidence",  "H7_same_category_evidence",  None),
]

TARGETS = ["medape", "medae"]
METRIC_DISPLAY = {"medape": "ΔMedAPE", "medae": "ΔMedAE"}


def stored_headline_test(data, label):
    """Return the stored headline significance summary for each ablation."""
    summary = data.get("wilcoxon", {})
    if label == "H5":
        summary = summary.get("0.85", {})
    return {
        m: {
            "median_delta": summary.get(m, {}).get("median_delta", float("nan")),
            "p_value": summary.get(m, {}).get("p_value", float("nan")),
            "n": summary.get(m, {}).get("n", 0),
        }
        for m in TARGETS
    }


def main():
    print("=" * 110)
    print("Top block — Correlational hypotheses (univariate Spearman ρ)")
    print("=" * 110)
    print(f"{'H':4s} {'Name':28s} {'Feature':18s}" + "".join(f"  {t:>16s}" for t in TARGETS))
    print("-" * 110)
    for label, name, dirname, feature in CORR_HYPOTHESES:
        # Look in either analyze.py output (results.json) OR analyze.json
        p1 = os.path.join(HERE, dirname, "results.json")
        if not os.path.exists(p1):
            print(f"{label:4s} {name:28s} {feature:18s}  [missing {p1}]")
            continue
        data = load_json(p1)
        spearman = data.get("univariate_spearman", {})
        cells = []
        for t in TARGETS:
            s = spearman.get(t, {}).get(feature, {})
            rho = s.get("rho", float("nan")); p = s.get("p", float("nan"))
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            cells.append(f"{rho:+.3f}{star} ({p:.0e})")
        print(f"{label:4s} {name:28s} {feature:18s}" + "".join(f"  {c:>16s}" for c in cells))

    print("\n" + "=" * 110)
    print("Bottom block — Ablation hypotheses (paired Wilcoxon Δ vs baseline; one Δ per benchmark)")
    print("=" * 110)
    print(f"{'H':4s} {'Name':28s} {'n':>4s}" + "".join(f"  {METRIC_DISPLAY[t]:>16s}" for t in TARGETS))
    print("-" * 110)
    for label, name, dirname, thr in ABL_HYPOTHESES:
        p = os.path.join(HERE, dirname, "ablation_results.json")
        if not os.path.exists(p):
            print(f"{label:4s} {name:28s}  [missing {p}]")
            continue
        data = load_json(p)
        wilc = stored_headline_test(data, label)
        n_show = wilc.get("medape", {}).get("n", 0)
        cells = []
        for t in TARGETS:
            v = wilc.get(t, {})
            md = v.get("median_delta", float("nan")); pv = v.get("p_value", float("nan"))
            star = "**" if (pv == pv and pv < 0.01) else ("*" if (pv == pv and pv < 0.05) else "")
            cells.append(f"{md:+.3f}{star} ({pv:.0e})")
        print(f"{label:4s} {name:28s} {n_show:>4d}" + "".join(f"  {c:>16s}" for c in cells))

    print("\nNote: * p<0.05, ** p<0.01.  Bottom block reads each ablation's stored headline Wilcoxon summary. H5 reports threshold=0.85.")


if __name__ == "__main__":
    main()
