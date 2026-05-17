#!/usr/bin/env python3
"""Score-matrix figures (§3.2) — produces all five panels of fig:data_distribution.

Outputs (in ``figures/``):
    bp_releases          — model releases over time
    bp_coverage          — coverage by release quarter
    bp_bench_mix         — benchmark mix by category
    bp_obs_concentrate   — observed cells by category
    bp_source_provenance — pie of source categories
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchpress.build_benchmark_matrix import load_score_matrix
from benchpress.build_benchmark_matrix.build_benchmark_matrix import (
    BENCHMARK_FILL_RULES,
    DEFAULT_EXCLUDED_BENCHMARKS,
    DEFAULT_EXCLUDED_MODELS,
    MODEL_FILL_RULES,
    _apply_benchmark_fill_rules,
    _apply_canonical_rules,
    _apply_model_fill_rules,
    _filter_status,
    _load_raw,
)
from benchpress.plot_helpers import style as S
from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET,
    CHARCOAL,
    CYAN_TEAL,
    GRAY,
    MEMENTO_MAGENTA,
    ROSE,
    LAVENDER,
    AQUA,
    SOL_BASE01,
    SOL_BASE1,
    SKY_BLUE,
    VANILLA_BLUE,
)

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def _save(fig, stem: str, *, pad_inches: float = 0.06) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
        print(path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Panels 1–4: data-distribution bars
# ──────────────────────────────────────────────────────────────────────────────

CAT_MAP = {
    "Agentic": "Agentic / tool use",
    "Agentic search": "Agentic / tool use",
    "Search Agent": "Agentic / tool use",
    "Tool Use": "Agentic / tool use",
    "Tool use": "Agentic / tool use",
    "Coding": "Coding",
    "Repository Code": "Coding",
    "Math": "Math",
    "Math/Vision": "Math",
    "Multimodal": "Multimodal / vision",
    "Vision": "Multimodal / vision",
    "Video/Multimodal": "Multimodal / vision",
    "Long Context": "Long context",
    "Long-context": "Long context",
    "Instruction Following": "Instruction following",
    "Instruction following": "Instruction following",
    "Knowledge": "Knowledge / QA",
    "QA": "Knowledge / QA",
    "Chinese": "Knowledge / QA",
    "Reasoning": "Reasoning",
    "Reasoning & Knowledge": "Reasoning",
    "Science": "Science",
    "Hallucination": "Hallucination / factuality",
    "Factuality": "Hallucination / factuality",
    "Safety": "Behavior / safety",
    "Behavior": "Behavior / safety",
    "Composite": "Composite / preference",
    "Human Preference": "Composite / preference",
    "Chat": "Composite / preference",
}

CAT_COLORS = {
    "Math": ANSWER_VIOLET,
    "Coding": CYAN_TEAL,
    "Agentic / tool use": ROSE,
    "Multimodal / vision": SKY_BLUE,
    "Instruction following": LAVENDER,
    "Long context": SOL_BASE01,
    "Knowledge / QA": VANILLA_BLUE,
    "Reasoning": MEMENTO_MAGENTA,
    "Hallucination / factuality": ROSE,
    "Science": ANSWER_VIOLET,
    "Behavior / safety": SOL_BASE1,
    "Composite / preference": AQUA,
}

DATA_DIST_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def _label_bars(ax, values, *, xpad: float = 0.5) -> None:
    for i, val in enumerate(values):
        ax.text(val + xpad, i, f"{int(val)}", va="center", ha="left",
                fontsize=11, color=CHARCOAL)


def make_data_distribution_panels() -> None:
    matrix, info, models, benches = load_score_matrix(
        return_info=True, return_metadata=True)
    models = models.copy()
    benches = benches.copy()

    models["release_date"] = pd.to_datetime(models["release_date"], errors="coerce")
    models["release_quarter"] = models["release_date"].dt.to_period("Q").astype(str)
    benches["group"] = benches["category"].map(CAT_MAP).fillna(benches["category"])

    obs = matrix.notna()
    models = models.join(obs.sum(axis=1).rename("observed_benchmarks"))
    benches = benches.join(obs.sum(axis=0).rename("observed_models"))

    with plt.rc_context(DATA_DIST_RC):
        # Panel 1: Model releases over time
        by_quarter = models.dropna(subset=["release_date"]).groupby("release_quarter").size()
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.bar(range(len(by_quarter)), by_quarter.values, color=MEMENTO_MAGENTA, width=0.78)
        ax.set_xticks(range(len(by_quarter)))
        ax.set_xticklabels(by_quarter.index, rotation=45, ha="right")
        ax.set_ylabel("# models")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _save(fig, "bp_releases")

        # Panel 2: Coverage by release quarter
        coverage = (
            models.dropna(subset=["release_date"])
            .groupby("release_quarter")["observed_benchmarks"]
            .agg(["median", "mean"])
            .reindex(by_quarter.index)
        )
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.plot(range(len(coverage)), coverage["median"], "o-", color=MEMENTO_MAGENTA,
                linewidth=2.5, label="median")
        ax.plot(range(len(coverage)), coverage["mean"], "s--", color=VANILLA_BLUE,
                linewidth=2.0, label="mean")
        ax.set_xticks(range(len(coverage)))
        ax.set_xticklabels(coverage.index, rotation=45, ha="right")
        ax.set_ylabel("Observed benchmarks per model")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        _save(fig, "bp_coverage")

        # Panel 3: Benchmark category mix
        cat_counts = benches["group"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.barh(cat_counts.index, cat_counts.values,
                color=[CAT_COLORS.get(c, GRAY) for c in cat_counts.index])
        _label_bars(ax, cat_counts.values)
        ax.set_xlabel("# benchmarks")
        ax.set_xlim(0, max(cat_counts.values) + 5)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        _save(fig, "bp_bench_mix")

        # Panel 4: Observed cells by category
        obs_by_cat = benches.groupby("group")["observed_models"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.barh(obs_by_cat.index, obs_by_cat.values,
                color=[CAT_COLORS.get(c, GRAY) for c in obs_by_cat.index])
        _label_bars(ax, obs_by_cat.values, xpad=6)
        ax.set_xlabel("# observed cells")
        ax.set_xlim(0, max(obs_by_cat.values) + 80)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        _save(fig, "bp_obs_concentrate")

    print(
        f"\n[caption text] Current filtered matrix: {info.n_models} models "
        f"\\times {info.n_benchmarks} benchmarks, {info.n_observations:,} observed cells "
        f"({100 * info.fill_rate:.1f}\\% fill)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Panel 5: source-provenance pie
# ──────────────────────────────────────────────────────────────────────────────

PRIORITY = {
    "official_blog": 1,
    "model_card": 2,
    "tech_report": 3,
    "leaderboard": 4,
    "official_paper": 4,
    "academic_paper": 5,
    "third_party_aggregator": 6,
    "third_party": 7,
    "unknown": 8,
    "": 9,
}

GROUPS = {
    "official_blog": "Model provider",
    "tech_report": "Model provider",
    "model_card": "Model provider",
    "official_paper": "Model provider",
    "leaderboard": "Benchmark leaderboard",
    "third_party_aggregator": "Third-party",
    "third_party": "Third-party",
    "academic_paper": "Third-party",
    "unknown": "Third-party",
    "": "Third-party",
}


def _compute_provenance() -> Counter:
    sm = load_score_matrix()
    adopted_models = set(sm.index)
    adopted_benchmarks = set(sm.columns)
    adopted_cells = {
        (mid, bid)
        for mid in sm.index
        for bid in sm.columns
        if not np.isnan(sm.at[mid, bid])
    }

    scores = _filter_status(_load_raw()["scores"], ("verified", "verified_third_party"))
    excluded_models = set(DEFAULT_EXCLUDED_MODELS)
    excluded_benchmarks = set(DEFAULT_EXCLUDED_BENCHMARKS)
    scores = _apply_benchmark_fill_rules(
        scores, fill_rules=BENCHMARK_FILL_RULES, exclude_models=excluded_models)
    scores = _apply_model_fill_rules(
        scores, fill_rules=MODEL_FILL_RULES, exclude_benchmarks=excluded_benchmarks)
    scores = _apply_canonical_rules(
        scores, exclude_models=excluded_models,
        exclude_benchmarks=excluded_benchmarks)

    cell_best: dict = {}
    for s in scores:
        mid, bid = s["model_id"], s["benchmark_id"]
        if (mid in adopted_models and bid in adopted_benchmarks
                and (mid, bid) in adopted_cells):
            status = s.get("audit_status", "")
            if status in ("verified", "verified_third_party"):
                st = s.get("source_type", "unknown")
                p = PRIORITY.get(st, 10)
                key = (mid, bid)
                if key not in cell_best or p < cell_best[key][1]:
                    cell_best[key] = (st, p)

    group_counts: Counter = Counter()
    for (_mid, _bid), (st, _p) in cell_best.items():
        group_counts[GROUPS.get(st, "Third-party")] += 1
    return group_counts


def make_source_provenance_panel() -> None:
    print("[source_provenance] computing...")
    S.apply_single()

    counts = _compute_provenance()
    total = sum(counts.values())
    print(f"    total cells: {total}")

    labels = ["Model provider", "Benchmark leaderboard", "Third-party"]
    sizes = [counts[l] for l in labels]
    colors = [S.MEMENTO_MAGENTA, S.VANILLA_BLUE, S.CYAN_TEAL]
    explode = (0.02, 0.02, 0.02)

    fig, ax = plt.subplots(figsize=(2.0, 1.95))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.30)
    wedges, _texts, _autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors, explode=explode,
        startangle=90, pctdistance=0.65, radius=1.12,
        textprops={"fontsize": 9, "color": "white", "fontweight": "bold"},
    )
    ax.set_aspect("equal")

    legend_labels = [f"{l} ({s:,})" for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="upper center",
              bbox_to_anchor=(0.5, 0.055), fontsize=10,
              frameon=False, handlelength=1.0, handletextpad=0.35,
              labelspacing=0.15, borderaxespad=0.0, ncol=1)

    _save(fig, "bp_source_provenance", pad_inches=0.01)
    for l, s in zip(labels, sizes):
        print(f"    {l}: {s} ({100 * s / total:.1f}%)")


def main() -> None:
    make_data_distribution_panels()
    make_source_provenance_panel()


if __name__ == "__main__":
    main()
