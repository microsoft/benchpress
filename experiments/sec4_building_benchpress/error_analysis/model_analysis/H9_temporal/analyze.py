#!/usr/bin/env python3
"""Rolling temporal robustness: split models into thirds by release date.

Group A (oldest 33%) and Group B (middle 33%) each serve as the sole
training set.  Group C (newest 33%) is always the evaluation target.

For each target model in C, reveal k benchmarks, predict the rest.
Compare condition A vs condition B via paired Wilcoxon signed-rank test.

Tests: "Does having temporally closer training data help predict new models?"

Output: results.json + bp_rolling_temporal.pdf
"""
import os
import time

import numpy as np
from benchpress.shard_utils import short_text_hash
from benchpress.stats import paired_wilcoxon
from benchpress.io_utils import load_json, write_json

from benchpress.all_methods import (
    MODEL_IDS, MODEL_NAMES, BENCH_IDS, BENCH_NAMES,
    M_FULL, N_MODELS, N_BENCH, predict_benchpress_scores,
)
from benchpress.build_benchmark_matrix import MODELS
from benchpress.evaluation_harness import compute_prediction_error

HERE = os.path.dirname(os.path.abspath(__file__))
OBSERVED = ~np.isnan(M_FULL)
MODEL_IDX = {mid: i for i, mid in enumerate(MODEL_IDS)}

# ── Model release dates ──────────────────────────────────────────────

MODEL_DATES = {}
for m in MODELS:
    mid, _name, _prov, rd = m[0], m[1], m[2], m[3]
    if rd and mid in MODEL_IDX:
        MODEL_DATES[mid] = rd

# Sort all models by release date
ALL_SORTED = sorted(MODEL_DATES.items(), key=lambda x: x[1])
N_TOTAL = len(ALL_SORTED)
CUT1 = N_TOTAL // 3
CUT2 = 2 * N_TOTAL // 3

GROUP_A_MIDS = [mid for mid, _ in ALL_SORTED[:CUT1]]       # oldest
GROUP_B_MIDS = [mid for mid, _ in ALL_SORTED[CUT1:CUT2]]   # middle
GROUP_C_MIDS = [mid for mid, _ in ALL_SORTED[CUT2:]]       # newest = eval

K_VALUES = [1, 3, 5, 8, 10, 15]
N_SEEDS = 10
METRICS = ["medape", "medae"]


def run_condition(train_mids, target_mids, condition_name):
    """Run BenchPress with given training models, predicting target models."""
    print(f"\n  Condition: {condition_name}")
    print(f"    Train: {len(train_mids)} models, Target: {len(target_mids)} models")
    # Stable condition-specific offset keeps random streams separate across
    # temporal baselines while remaining reproducible across Python processes.
    condition_offset = int(short_text_hash(condition_name, n=8, algorithm="md5"), 16) % 10000

    # Per-seed aggregate metrics
    seed_metrics = {k: {m: [] for m in METRICS} for k in K_VALUES}
    # Per-seed per-model metrics (for paired comparison)
    seed_per_model = {k: [] for k in K_VALUES}
    # Raw predictions
    raw_preds = {k: {"seeds": [], "models": [], "benchmarks": [],
                      "actuals": [], "preds": []} for k in K_VALUES}

    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed * 500 + condition_offset)

        for k in K_VALUES:
            M_train = np.full((N_MODELS, N_BENCH), np.nan)

            # Fill training models fully
            for mid in train_mids:
                i = MODEL_IDX[mid]
                for j in range(N_BENCH):
                    if OBSERVED[i, j]:
                        M_train[i, j] = M_FULL[i, j]

            # For each target: reveal k benchmarks, hide rest
            test_set = []
            per_model_test = {}  # model_idx -> list of (i, j)
            for mid in target_mids:
                i = MODEL_IDX[mid]
                obs_j = np.where(OBSERVED[i])[0]
                if len(obs_j) < k + 2:
                    continue
                obs_shuffled = obs_j.copy()
                rng.shuffle(obs_shuffled)
                revealed = obs_shuffled[:k]
                hidden = obs_shuffled[k:]

                for j in revealed:
                    M_train[i, j] = M_FULL[i, j]
                per_model_test[i] = []
                for j in hidden:
                    test_set.append((i, j))
                    per_model_test[i].append((i, j))

            if len(test_set) < 3:
                continue

            M_pred = predict_benchpress_scores(M_train)

            # Aggregate metrics
            actual = np.array([M_FULL[i, j] for i, j in test_set], dtype=float)
            predicted = np.array([M_pred[i, j] for i, j in test_set], dtype=float)
            m = compute_prediction_error(actual, predicted)
            for metric in METRICS:
                val = float(m[metric])
                if np.isfinite(val):
                    seed_metrics[k][metric].append(val)

            # Per-model metrics (for paired test)
            model_results = {}
            for model_i, cells in per_model_test.items():
                if not cells:
                    continue
                a = np.array([M_FULL[i, j] for i, j in cells], dtype=float)
                p = np.array([M_pred[i, j] for i, j in cells], dtype=float)
                pm = compute_prediction_error(a, p)
                model_results[model_i] = {
                    metric: float(pm[metric]) for metric in METRICS
                }
            seed_per_model[k].append(model_results)

            # Raw predictions
            for i, j in test_set:
                a, p = M_FULL[i, j], M_pred[i, j]
                if np.isfinite(a) and np.isfinite(p):
                    raw_preds[k]["seeds"].append(seed)
                    raw_preds[k]["models"].append(i)
                    raw_preds[k]["benchmarks"].append(int(j))
                    raw_preds[k]["actuals"].append(round(float(a), 6))
                    raw_preds[k]["preds"].append(round(float(p), 6))

        if (seed + 1) % 5 == 0:
            print(f"    seed {seed+1}/{N_SEEDS}")

    # Summarize
    summary = {}
    for k in K_VALUES:
        summary[str(k)] = {}
        for metric in METRICS:
            vals = seed_metrics[k][metric]
            summary[str(k)][metric] = {
                "median": float(np.median(vals)) if vals else float("nan"),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if vals else 0,
                "n": len(vals),
                "values": [round(v, 6) for v in vals],
            }
    return summary, seed_per_model, raw_preds


def run_rolling_temporal():
    print("[Rolling Temporal: A vs B design]")
    print(f"  Group A (oldest):  {len(GROUP_A_MIDS)} models")
    print(f"  Group B (middle):  {len(GROUP_B_MIDS)} models")
    print(f"  Group C (target):  {len(GROUP_C_MIDS)} models")
    t0 = time.time()

    # Run both conditions
    sum_a, per_model_a, raw_a = run_condition(
        GROUP_A_MIDS, GROUP_C_MIDS, "A_oldest")
    print(f"  Condition A done ({time.time()-t0:.0f}s)")

    sum_b, per_model_b, raw_b = run_condition(
        GROUP_B_MIDS, GROUP_C_MIDS, "B_middle")
    print(f"  Condition B done ({time.time()-t0:.0f}s)")

    # Paired comparison: A vs B
    print("\n[Paired Wilcoxon: A vs B]")
    comparison = {}
    for k in K_VALUES:
        comparison[str(k)] = {}
        for metric in METRICS:
            va = sum_a[str(k)][metric]["values"]
            vb = sum_b[str(k)][metric]["values"]
            n = min(len(va), len(vb))
            if n < 3:
                comparison[str(k)][metric] = {
                    "median_diff": float("nan"),
                    "p_value": float("nan"),
                    "n": n,
                }
                continue
            md, p = paired_wilcoxon(va[:n], vb[:n])
            comparison[str(k)][metric] = {
                "median_diff": round(md, 6),
                "p_value": round(p, 6),
                "n": n,
                "median_A": round(float(np.median(va[:n])), 6),
                "median_B": round(float(np.median(vb[:n])), 6),
            }
            print(f"  k={k:2d} {metric:20s}: "
                  f"A={np.median(va[:n]):.4f} B={np.median(vb[:n]):.4f} "
                  f"Δ(A-B)={md:+.4f} p={p:.4f}")

    results = {
        "design": "A_vs_B_thirds",
        "group_A": {
            "description": "oldest third",
            "n_models": len(GROUP_A_MIDS),
            "models": GROUP_A_MIDS,
            "date_range": [ALL_SORTED[0][1], ALL_SORTED[CUT1-1][1]],
        },
        "group_B": {
            "description": "middle third",
            "n_models": len(GROUP_B_MIDS),
            "models": GROUP_B_MIDS,
            "date_range": [ALL_SORTED[CUT1][1], ALL_SORTED[CUT2-1][1]],
        },
        "group_C": {
            "description": "newest third (eval targets)",
            "n_models": len(GROUP_C_MIDS),
            "models": GROUP_C_MIDS,
            "date_range": [ALL_SORTED[CUT2][1], ALL_SORTED[-1][1]],
        },
        "k_values": K_VALUES,
        "n_seeds": N_SEEDS,
        "condition_A": sum_a,
        "condition_B": sum_b,
        "comparison_A_vs_B": comparison,
        "raw_predictions_A": {str(k): raw_a[k] for k in K_VALUES},
        "raw_predictions_B": {str(k): raw_b[k] for k in K_VALUES},
    }

    out_path = os.path.join(HERE, "results.json")
    write_json(out_path, results, indent=2, default=str)
    print(f"\n[saved] {out_path} ({time.time()-t0:.0f}s)")

    # Print summary table
    print("\n=== Summary (k=10) ===")
    k = "10"
    print(f"{'Metric':<20} {'A (oldest)':<12} {'B (middle)':<12} {'Δ(A-B)':<12} {'p-value':<10}")
    print("-" * 66)
    for metric in METRICS:
        c = comparison[k][metric]
        ma = c.get("median_A", float("nan"))
        mb = c.get("median_B", float("nan"))
        md = c.get("median_diff", float("nan"))
        p = c.get("p_value", float("nan"))
        print(f"{metric:<20} {ma:<12.4f} {mb:<12.4f} {md:<+12.4f} {p:<10.4f}")

    return results


def plot_figure(results=None):
    """Line plot: MedAPE for A (oldest) vs B (middle) across k values."""
    import matplotlib.pyplot as plt
    from benchpress.plot_helpers.visual_identity import (
        CHARCOAL, PEACH, apply_single, save_fig,
    )
    apply_single()

    if results is None:
        results = load_json(os.path.join(HERE, "results.json"))

    k_plot = [1, 3, 5, 10]
    comp = results["comparison_A_vs_B"]

    a_vals = [comp[str(k)]["medape"]["median_A"] for k in k_plot]
    b_vals = [comp[str(k)]["medape"]["median_B"] for k in k_plot]

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(k_plot, a_vals, 'o-', color=CHARCOAL, label='Oldest third (A)',
            markersize=9, zorder=3)
    ax.plot(k_plot, b_vals, 's-', color=PEACH, label='Middle third (B)',
            markersize=9, zorder=3)

    # Mark significant differences
    for k in k_plot:
        p = comp[str(k)]["medape"]["p_value"]
        if p < 0.05:
            y_top = max(comp[str(k)]["medape"]["median_A"],
                        comp[str(k)]["medape"]["median_B"])
            ax.annotate('*', xy=(k, y_top), fontsize=18, ha='center',
                        va='bottom', fontweight='bold', color=CHARCOAL)

    ax.set_xlabel('Revealed benchmarks ($k$)')
    ax.set_ylabel('MedAPE (%)')
    ax.set_xticks(k_plot)
    ax.legend(frameon=False)
    ax.set_xlim(0.2, 11)

    save_fig('bp_rolling_temporal')


if __name__ == "__main__":
    results = run_rolling_temporal()
    plot_figure(results)
