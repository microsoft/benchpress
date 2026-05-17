#!/usr/bin/env python3
"""Render the current arXiv Hero Figure panels from stored result summaries."""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from benchpress.io_utils import load_json
from benchpress.plot_helpers.visual_identity import (
    ANSWER_VIOLET as VIOLET,
    CHARCOAL,
    CYAN_TEAL as TEAL,
    GRAY,
    MEMENTO_MAGENTA as MAGENTA,
    ROSE,
    SOL_BASE01 as MUTED,
    SOL_BASE2 as GRID,
    VANILLA_BLUE as BLUE,
)


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

SUMMARY_PATH = RESULTS_DIR / "hero_candidate_grid_summary.json"
PHI_SUMMARY_PATH = RESULTS_DIR / "phi4_reasoning_plus_gpqa_keepk_summary.json"
RANDOM_PATH = (
    HERE / ".." / ".." / "sec5_findings" / "optimal_probe" / "results"
    / "random_medape_hero_all_known.json.gz"
).resolve()
GREEDY_MEDAE_PATH = (
    HERE / ".." / ".." / "sec5_findings" / "optimal_probe" / "results"
    / "greedy_medae_targets_tall_candidates_tall.json.gz"
).resolve()
GREEDY_MEDAE_COST_AWARE_PATH = (
    HERE / ".." / ".." / "sec5_findings" / "optimal_probe" / "results"
    / "greedy_medae_targets_tall_candidates_usercheap.json.gz"
).resolve()
RANK_GREEDY_PATH = (
    HERE / ".." / ".." / "sec5_findings" / "ranking_preservation"
    / "greedy_probe_set" / "results"
    / "greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz"
).resolve()
RANK_GREEDY_COST_AWARE_PATH = (
    HERE / ".." / ".." / "sec5_findings" / "ranking_preservation"
    / "greedy_probe_set" / "results"
    / "greedy_pairwise_margin5_top10_targets_usercheap_candidates_usercheap.json.gz"
).resolve()

PICKS = [
    ("gpt-5.5", "browsecomp"),
    ("claude-opus-4.7", "hle"),
    ("phi-4-reasoning-plus", "gpqa_diamond"),
    ("deepseek-v4-pro", "terminal_bench"),
]
DISPLAY = {
    ("gpt-5.5", "browsecomp"): ("GPT-5.5", "BrowseComp"),
    ("claude-opus-4.7", "hle"): ("Claude Opus 4.7", "HLE"),
    ("phi-4-reasoning-plus", "gpqa_diamond"): (
        "Phi-4 Reasoning Plus",
        "GPQA Diamond",
    ),
    ("deepseek-v4-pro", "terminal_bench"): (
        "DeepSeek V4 Pro",
        "Terminal-Bench 2.0",
    ),
}
PANEL_A_COLORS = [MAGENTA, BLUE, VIOLET, TEAL]
PDF_CREATION_DATES = {
    "bp_hero_panel_a_examples.pdf": "D:20260505184314-07'00'",
    "bp_hero_panel_b_overall.pdf": "D:20260505190328-07'00'",
    "bp_ranking_preservation_overall.pdf": "D:20260506192400-07'00'",
}

def save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"CreationDate": PDF_CREATION_DATES[path.name]}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Bad value for infodict keyword CreationDate")
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def selected_examples():
    candidate = load_json(SUMMARY_PATH)["summary"]
    by_pair = {(x["model_id"], x["bench_id"]): x for x in candidate}

    phi = load_json(PHI_SUMMARY_PATH)
    cfg = phi["config"]
    by_pair[(cfg["model_id"], cfg["bench_id"])] = {
        "model_id": cfg["model_id"],
        "model": DISPLAY[(cfg["model_id"], cfg["bench_id"])][0],
        "bench_id": cfg["bench_id"],
        "benchmark": cfg["bench_name"],
        "category": "Science",
        "actual": cfg["target_score"],
        "baseline_ae": cfg["baseline_benchmark_median_ae"],
        "random": [
            {
                "k": r["k"],
                "median": r["median_ae"],
                "q1": r["q1_ae"],
                "q3": r["q3_ae"],
                "revealed_count": r["revealed_count"],
            }
            for r in phi["summary"]
        ],
        "cost_unaware": [],
        "cost_aware": [],
        "score_for_display": cfg["target_score"],
    }

    missing = [pair for pair in PICKS if pair not in by_pair]
    if missing:
        raise RuntimeError(f"Missing selected hero pairs: {missing}")
    return [by_pair[pair] for pair in PICKS]


def probe_policy_curves():
    random = load_json(RANDOM_PATH)
    greedy = load_json(GREEDY_MEDAE_PATH)
    greedy_cost_aware = load_json(GREEDY_MEDAE_COST_AWARE_PATH)
    rank_greedy = load_json(RANK_GREEDY_PATH)
    rank_greedy_cost_aware = load_json(RANK_GREEDY_COST_AWARE_PATH)

    random_by_k = defaultdict(list)
    for row in random["summary_by_k_seed"]:
        k = int(row["k"])
        if 1 <= k <= 10:
            random_by_k[k].append(row)

    random_k = np.arange(1, 11)
    random_medae = np.array([
        np.median([r["medae"] for r in random_by_k[k]])
        for k in random_k
    ])
    random_q1 = np.array([
        np.percentile([r["medae"] for r in random_by_k[k]], 25)
        for k in random_k
    ])
    random_q3 = np.array([
        np.percentile([r["medae"] for r in random_by_k[k]], 75)
        for k in random_k
    ])
    greedy_k = np.array([
        int(s["step"]) for s in greedy["trajectory"]
        if 1 <= int(s["step"]) <= 10
    ])
    greedy_medae = np.array([
        float(s["medae"]) for s in greedy["trajectory"]
        if 1 <= int(s["step"]) <= 10
    ])
    greedy_cost_aware_k = np.array([
        int(s["step"]) for s in greedy_cost_aware["trajectory"]
        if 1 <= int(s["step"]) <= 10
    ])
    greedy_cost_aware_medae = np.array([
        float(s["medae"]) for s in greedy_cost_aware["trajectory"]
        if 1 <= int(s["step"]) <= 10
    ])
    rank_k = np.array([
        0,
        *[
            int(s["step"]) for s in rank_greedy["trajectory"]
            if 1 <= int(s["step"]) <= 10
        ],
    ])
    rank_acc = np.array([
        0.0,
        *[
            100 * float(s["pairwise_accuracy_margin5"])
            for s in rank_greedy["trajectory"]
            if 1 <= int(s["step"]) <= 10
        ],
    ])
    rank_cost_aware_k = np.array([
        0,
        *[
            int(s["step"]) for s in rank_greedy_cost_aware["trajectory"]
            if 1 <= int(s["step"]) <= 10
        ],
    ])
    rank_cost_aware_acc = np.array([
        0.0,
        *[
            100 * float(s["pairwise_accuracy_margin5"])
            for s in rank_greedy_cost_aware["trajectory"]
            if 1 <= int(s["step"]) <= 10
        ],
    ])
    rand_rank_k, rand_rank_acc, rand_rank_q1, rand_rank_q3 = ranking_data_from_raw(random["raw_predictions"])

    # k=0 baseline: predict every observed cell with its benchmark column median
    from benchpress.evaluation_harness import M_FULL, OBSERVED, compute_prediction_error
    from benchpress.all_methods import predict_benchmark_median_scores
    M_pred_baseline = predict_benchmark_median_scores(M_FULL)
    test_cells = list(zip(*np.where(OBSERVED)))
    baseline_metrics = compute_prediction_error(
        M_FULL, M_pred_baseline, test_set=test_cells, aggregation='pool')
    baseline_medae = float(baseline_metrics['medae'])

    return {
        "random_k": random_k,
        "random_medae": random_medae,
        "random_q1": random_q1,
        "random_q3": random_q3,
        "greedy_k": greedy_k,
        "greedy_medae": greedy_medae,
        "greedy_cost_aware_k": greedy_cost_aware_k,
        "greedy_cost_aware_medae": greedy_cost_aware_medae,
        "baseline_medae": baseline_medae,
        "rand_rank_k": rand_rank_k,
        "rand_rank_acc": rand_rank_acc,
        "rand_rank_q1": rand_rank_q1,
        "rand_rank_q3": rand_rank_q3,
        "rank_k": rank_k,
        "rank_acc": rank_acc,
        "rank_cost_aware_k": rank_cost_aware_k,
        "rank_cost_aware_acc": rank_cost_aware_acc,
        "greedy_trajectory": greedy["trajectory"],
        "greedy_cost_aware_trajectory": greedy_cost_aware["trajectory"],
    }


def ranking_data_from_raw(raw_predictions, margin=5.0, k_max=10):
    grouped = defaultdict(list)
    for row in raw_predictions:
        k = int(row["k"])
        if 1 <= k <= k_max:
            grouped[(k, int(row["seed"]), int(row["bench"]))].append(row)

    by_k = defaultdict(list)
    for (k, _seed, _bench), rows in grouped.items():
        if len(rows) < 2:
            continue
        actual = np.array([float(r["actual"]) for r in rows])
        pred = np.array([float(r["pred"]) for r in rows])
        true_diffs = actual[:, None] - actual[None, :]
        pred_diffs = pred[:, None] - pred[None, :]
        upper = np.triu(np.ones(true_diffs.shape, dtype=bool), k=1)
        comparable = upper & (np.abs(true_diffs) >= margin) & (true_diffs != 0)
        n_pairs = int(comparable.sum())
        if n_pairs == 0:
            continue
        correct = int(
            (np.sign(true_diffs[comparable]) == np.sign(pred_diffs[comparable])).sum()
        )
        by_k[k].append(correct / n_pairs)

    return (
        np.array([0, *range(1, k_max + 1)]),
        np.array([0.0, *[100 * float(np.median(by_k[k])) for k in range(1, k_max + 1)]]),
        np.array([0.0, *[100 * float(np.percentile(by_k[k], 25)) for k in range(1, k_max + 1)]]),
        np.array([0.0, *[100 * float(np.percentile(by_k[k], 75)) for k in range(1, k_max + 1)]]),
    )


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
    })


def render_panel_a(selected) -> Path:
    fig_a, axes_a = plt.subplots(2, 2, figsize=(6.0, 5.7))
    axes_a = axes_a.ravel()
    for idx, (ax, item) in enumerate(zip(axes_a, selected)):
        line_color = PANEL_A_COLORS[idx % len(PANEL_A_COLORS)]
        pair = (item["model_id"], item["bench_id"])
        model_name, bench_name = DISPLAY[pair]
        vals = [v for v in item["random"] if 1 <= int(v["k"]) <= 10]
        x = np.array([0] + [v["k"] for v in vals])
        base = float(item["baseline_ae"])
        med = np.array([base] + [float(v["median"]) for v in vals])
        q1 = np.array([base] + [float(v["q1"]) for v in vals])
        q3 = np.array([base] + [float(v["q3"]) for v in vals])

        ax.plot(x, med, "o-", color=line_color, lw=1.65, ms=3.8, zorder=3)
        ax.fill_between(x, q1, q3, color=line_color, alpha=0.14, lw=0)
        ax.plot(
            [0], [base], marker="D", color="white", markeredgecolor=CHARCOAL,
            markeredgewidth=1.0, ms=4.8, zorder=4,
        )
        ax.annotate(
            "Benchmark median", (0, base), xytext=(8, 2),
            textcoords="offset points", fontsize=12.0, color=CHARCOAL,
            ha="left", va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-", color=CHARCOAL, lw=0.9, shrinkA=0, shrinkB=3,
            ),
        )
        ax.axhline(5, color=CHARCOAL, ls="--", lw=1.35, alpha=0.88)
        ax.axhline(2, color=MUTED, ls=":", lw=1.45, alpha=0.88)
        ax.set_xlim(-0.45, 10.45)
        ax.set_xticks(list(range(0, 11)))
        ax.set_ylim(0, 30)
        ax.text(
            0.22, 5, "5", fontsize=12.0, color=CHARCOAL, ha="left", va="center",
            bbox=dict(
                boxstyle="round,pad=0.08", facecolor="white",
                edgecolor="none", alpha=0.88,
            ),
        )
        ax.text(
            0.22, 2, "2", fontsize=12.0, color=MUTED, ha="left", va="center",
            bbox=dict(
                boxstyle="round,pad=0.08", facecolor="white",
                edgecolor="none", alpha=0.88,
            ),
        )
        ax.tick_params(labelsize=12.0, pad=1.3)
        if idx in (0, 2):
            ax.set_ylabel("Absolute error", fontsize=12.0)
        else:
            ax.set_yticklabels([])
        if idx in (2, 3):
            ax.set_xlabel("# Known benchmarks", fontsize=12.0, labelpad=1.3)
        ax.grid(axis="y", color=GRID, alpha=0.55, lw=0.6)
        ax.set_title(
            f"{model_name}\n{bench_name}", fontsize=12.2,
            fontweight="bold", color=CHARCOAL, pad=3,
        )

    fig_a.subplots_adjust(
        left=0.108, right=0.990, top=0.925, bottom=0.112,
        hspace=0.42, wspace=0.24,
    )
    output = FIGURES_DIR / "bp_hero_panel_a_examples.pdf"
    save_pdf(fig_a, output)
    return output


def render_panel_b(curves) -> Path:
    fig_b, ax = plt.subplots(1, 1, figsize=(7.6, 7.2))
    base = float(curves["baseline_medae"])

    random_x = np.concatenate([[0], curves["random_k"]])
    random_y = np.concatenate([[base], curves["random_medae"]])
    random_q1 = np.concatenate([[base], curves["random_q1"]])
    random_q3 = np.concatenate([[base], curves["random_q3"]])
    greedy_x = np.concatenate([[0], curves["greedy_k"]])
    greedy_y = np.concatenate([[base], curves["greedy_medae"]])
    cost_x = np.concatenate([[0], curves["greedy_cost_aware_k"]])
    cost_y = np.concatenate([[base], curves["greedy_cost_aware_medae"]])

    ax.fill_between(random_x, random_q1, random_q3, color=GRAY, alpha=0.14, lw=0)
    ax.plot(random_x, random_y, color=GRAY, lw=2.3, ls="--", marker="o", ms=5.5)
    ax.plot(greedy_x, greedy_y, color=MAGENTA, lw=2.5, ls="-", marker="o", ms=5.5)
    ax.plot(cost_x, cost_y, color=BLUE, lw=2.5, ls="-", marker="s", ms=5.2)
    ax.plot(
        [0], [base], marker="D", color="white", markeredgecolor=CHARCOAL,
        markeredgewidth=1.0, ms=5.6, zorder=5,
    )
    ax.annotate(
        "Benchmark median", (0, base), xytext=(8, -2),
        textcoords="offset points", fontsize=12.0, color=CHARCOAL,
        ha="left", va="top",
        bbox=dict(
            boxstyle="round,pad=0.18", facecolor="white",
            edgecolor="none", alpha=0.92,
        ),
        arrowprops=dict(
            arrowstyle="-", color=CHARCOAL, lw=0.9, shrinkA=0, shrinkB=3,
        ),
    )

    SHORT = {
        "HLE (Humanity's Last Exam)": "HLE",
        "Terminal-Bench 2.0": "Terminal-Bench",
        "Bullshit-Bench (Clear Pushback)": "Bullshit-Bench",
        "GDPval (Artificial Analysis ELO)": "GDPval",
        "Aider Polyglot (diff mode)": "Aider Polyglot",
        "\u03c4\u00b2-bench Airline": "\u03c4\u00b2-bench",
        "Codeforces Rating": "Codeforces",
    }
    def _short(n):
        return SHORT.get(n, n)

    greedy_names = [_short(s["added_benchmark_name"]) for s in curves["greedy_trajectory"][:10]]
    cost_names = [_short(s["added_benchmark_name"]) for s in curves["greedy_cost_aware_trajectory"][:10]]

    for i, name in enumerate(greedy_names, start=1):
        ax.annotate(
            name, xy=(i, curves["greedy_medae"][i-1]),
            xytext=(-3, -7), textcoords="offset points",
            fontsize=12, color=MAGENTA, ha="right", va="top",
            rotation=30,
            bbox=dict(
                boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.78,
            ),
        )
    for i, name in enumerate(cost_names, start=1):
        ax.annotate(
            name, xy=(i, curves["greedy_cost_aware_medae"][i-1]),
            xytext=(3, 7), textcoords="offset points",
            fontsize=12, color=BLUE, ha="left", va="bottom",
            rotation=30,
            bbox=dict(
                boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.78,
            ),
        )

    ax.set_xlim(-0.45, 10.45)
    ax.set_xticks(list(range(0, 11)))
    ax.set_ylim(bottom=1.5)
    ax.set_xlabel("# Top benchmarks", fontsize=18.0, labelpad=1.5)
    ax.set_ylabel("Median Absolute Error", fontsize=18.0)
    ax.set_title("Overall score prediction", fontsize=18.0, fontweight="bold", color=CHARCOAL, pad=3)
    ax.grid(axis="y", color=GRID, alpha=0.55, lw=0.6)
    ax.tick_params(labelsize=16.0, pad=1.5)

    handles = [
        Line2D([0], [0], color=GRAY, lw=1.55, linestyle="--", marker="o", markersize=5.2, label="Random benchmark set"),
        Line2D([0], [0], color=MAGENTA, lw=1.75, linestyle="-", marker="o", markersize=5.2, label="Most predictive benchmarks"),
        Line2D([0], [0], color=BLUE, lw=1.75, linestyle="-", marker="s", markersize=5.0, label="Low-cost benchmarks"),
    ]
    fig_b.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False,
        fontsize=14.0, bbox_to_anchor=(0.5, 0.012),
        handlelength=1.2, columnspacing=0.65,
        labelspacing=0.25, handletextpad=0.45,
    )
    fig_b.subplots_adjust(left=0.105, right=0.99, top=0.905, bottom=0.205)
    output = FIGURES_DIR / "bp_hero_panel_b_overall.pdf"
    save_pdf(fig_b, output)
    return output


def render_ranking_preservation_overall(curves) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.75))
    ax.fill_between(curves["rand_rank_k"], curves["rand_rank_q1"], curves["rand_rank_q3"], color=GRAY, alpha=0.14, lw=0, zorder=0)
    ax.plot(curves["rand_rank_k"], curves["rand_rank_acc"], color=GRAY, lw=2.6, ls="--", marker="o", ms=6.8, label="Random")
    ax.plot(curves["rank_k"], curves["rank_acc"], color=MAGENTA, lw=2.8, ls="-", marker="o", ms=6.8, label="Predictive")
    ax.plot(curves["rank_cost_aware_k"], curves["rank_cost_aware_acc"], color=BLUE, lw=2.8, ls="-", marker="s", ms=6.5, label="Low-cost")
    ax.set_xlim(-0.35, 10.45)
    ax.set_xticks([0, 5, 10])
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(["0%", "30%", "60%", "90%"])
    ax.set_ylabel("Pairwise accuracy", fontsize=20, labelpad=2.0)
    ax.set_xlabel("Top benchmarks", fontsize=20, labelpad=1.0)
    ax.grid(axis="y", color=GRID, alpha=0.55, lw=0.8)
    ax.tick_params(labelsize=18, pad=1.5)

    ax.legend(
        loc="lower right", frameon=False, fontsize=14,
        handlelength=1.0, labelspacing=0.12, handletextpad=0.25,
        borderaxespad=0.15,
    )
    fig.subplots_adjust(left=0.245, right=0.985, top=0.985, bottom=0.185)
    output = FIGURES_DIR / "bp_ranking_preservation_overall.pdf"
    save_pdf(fig, output)
    return output
    output = FIGURES_DIR / "bp_ranking_preservation_overall.pdf"
    save_pdf(fig, output)
    return output


def main() -> None:
    apply_style()
    selected = selected_examples()
    curves = probe_policy_curves()
    outputs = [
        render_panel_a(selected),
        render_panel_b(curves),
        render_ranking_preservation_overall(curves),
    ]
    for output in outputs:
        print(f"  -> {output.relative_to(HERE)}")


if __name__ == "__main__":
    main()
