#!/usr/bin/env python3
"""
Plot greedy probe selection results.

Usage:
    python plot.py
    python plot.py --in greedy_medape_targets_tall_candidates_usercheap.json.gz
    python plot.py --compare

Output: figures/<stem>.pdf and .png  (stem = input filename without .json).
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
from benchpress.plot_helpers.visual_identity import (
    apply_single, save_fig, integer_ticks,
    VANILLA_BLUE, PALETTE, GRAY,
    PROBE_COST_AWARE_STYLE, PROBE_COST_UNAWARE_STYLE, PROBE_RANDOM_STYLE,
)
from benchpress.evaluation_harness import (
    BENCH_IDS, BENCH_NAMES, OBSERVED, N_BENCH, N_MODELS, compute_prediction_error,
)
from benchpress.io_utils import load_json
import matplotlib.pyplot as plt
import numpy as np


SCORE_FIELD = {
    'medape': 'medape',
    'medae':  'medae',
}
YLABEL = {
    'medape': 'MedAPE (%)',
    'medae':  'MedAE',
}
VALUE_SUFFIX = {
    'medape': '%',
    'medae':  '',
}
GREEDY_PROTOCOL = 'all_known_probe_cells_zero_error_v1'
RANDOM_PROTOCOL = 'figure1_random_nested_probe_prefix_all_known_cells'
ALL_KNOWN_N = int(OBSERVED.sum())


def resolve_result_path(filename):
    in_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(in_path):
        alt = in_path[:-3] if in_path.endswith('.gz') else in_path + '.gz'
        if os.path.exists(alt):
            in_path = alt
    return in_path


def load_result(filename):
    in_path = resolve_result_path(filename)
    return os.path.basename(in_path), load_json(in_path)


def assert_greedy_result(data, label, metric):
    config = data.get('config', {})
    expected = {
        'metric': metric,
        'eval_protocol': GREEDY_PROTOCOL,
    }
    actual = {k: config.get(k) for k in expected}
    if actual != expected:
        raise ValueError(f"{label} config mismatch: expected {expected}, found {actual}")
    if int(config.get('n_target_cells', -1)) != ALL_KNOWN_N:
        raise ValueError(
            f"{label} denominator mismatch: expected n_target_cells={ALL_KNOWN_N}, "
            f"found {config.get('n_target_cells')}"
        )


def assert_random_result(data, label):
    config = data.get('config', {})
    k_max = int(config.get('k_max', -1))
    n_seeds = int(config.get('n_seeds', -1))
    if k_max < 10:
        raise ValueError(f"{label} random k_max must be at least 10, found {k_max}")
    expected = {
        'protocol': RANDOM_PROTOCOL,
        'n_seeds': 10,
        'model_limit': None,
        'n_models': N_MODELS,
        'n_bench': N_BENCH,
    }
    actual = {k: config.get(k) for k in expected}
    if actual != expected:
        raise ValueError(f"{label} config mismatch: expected {expected}, found {actual}")

    summary = data.get('summary_by_k_seed', [])
    expected_keys = {(k, seed) for k in range(1, k_max + 1) for seed in range(n_seeds)}
    actual_keys = {(int(r.get('k')), int(r.get('seed'))) for r in summary}
    if actual_keys != expected_keys:
        raise ValueError(
            f"{label} random summary grid mismatch: missing="
            f"{sorted(expected_keys - actual_keys)[:10]} extra="
            f"{sorted(actual_keys - expected_keys)[:10]}"
        )
    bad_n = sorted({int(r.get('n', -1)) for r in summary if int(r.get('n', -1)) != ALL_KNOWN_N})
    if bad_n:
        raise ValueError(f"{label} random denominator mismatch: expected {ALL_KNOWN_N}, found {bad_n}")
    expected_raw = ALL_KNOWN_N * k_max * n_seeds
    raw = data.get('raw_predictions', [])
    if len(raw) != expected_raw:
        raise ValueError(f"{label} raw row mismatch: expected {expected_raw}, found {len(raw)}")


def infer_stem(filename):
    return filename[:-len('.json.gz')] if filename.endswith('.json.gz') else os.path.splitext(filename)[0]


def trajectory_scores(data, metric):
    field = SCORE_FIELD[metric]
    trajectory = data['trajectory']
    budgets = np.array([t['step'] for t in trajectory])
    scores = np.array([float(t[field]) for t in trajectory])
    added = [t['added_benchmark_name'] for t in trajectory]
    return trajectory, budgets, scores, added


def random_scores(random_data, metric):
    if metric not in SCORE_FIELD:
        raise ValueError('Random baseline supports medape and medae metrics.')

    grouped = {}
    for row in random_data['raw_predictions']:
        key = (int(row['k']), int(row['seed']))
        grouped.setdefault(key, {'actual': [], 'pred': []})
        grouped[key]['actual'].append(float(row['actual']))
        grouped[key]['pred'].append(float(row['pred']))

    by_k = {}
    field = SCORE_FIELD[metric]
    for (k, _seed), vals in grouped.items():
        m = compute_prediction_error(
            np.array(vals['actual'], dtype=float),
            np.array(vals['pred'], dtype=float),
            aggregation='pool',
        )
        by_k.setdefault(k, []).append(float(m[field]))

    budgets = np.array(sorted(by_k))
    means = np.array([np.mean(by_k[k]) for k in budgets])
    p25 = np.array([np.percentile(by_k[k], 25) for k in budgets])
    p75 = np.array([np.percentile(by_k[k], 75) for k in budgets])
    return budgets, means, p25, p75


def plot_compare(all_data, cheap_data, metric, out_stem, random_data=None):
    _, budgets_all, scores_all, _ = trajectory_scores(all_data, metric)
    max_k = 10
    budgets_all = budgets_all[:max_k]
    scores_all = scores_all[:max_k]
    scores_cheap = None
    if cheap_data is not None:
        _, budgets_cheap, scores_cheap, _ = trajectory_scores(cheap_data, metric)
        budgets_cheap = budgets_cheap[:max_k]
        scores_cheap = scores_cheap[:max_k]

    # k=0 baseline: predict every observed cell with its benchmark column median
    from benchpress.evaluation_harness import M_FULL
    from benchpress.all_methods import predict_benchmark_median_scores
    M_pred_baseline = predict_benchmark_median_scores(M_FULL)
    test_cells = list(zip(*np.where(OBSERVED)))
    baseline_metrics = compute_prediction_error(
        M_FULL, M_pred_baseline, test_set=test_cells, aggregation='pool')
    baseline_score = float(baseline_metrics[metric])

    budgets_all = np.concatenate([[0], np.asarray(budgets_all)])
    scores_all = np.concatenate([[baseline_score], np.asarray(scores_all)])
    if scores_cheap is not None:
        budgets_cheap = np.concatenate([[0], np.asarray(budgets_cheap)])
        scores_cheap = np.concatenate([[baseline_score], np.asarray(scores_cheap)])

    apply_single()
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    all_style = PROBE_COST_UNAWARE_STYLE
    cheap_style = PROBE_COST_AWARE_STYLE
    random_style = PROBE_RANDOM_STYLE
    ax.plot(budgets_all, scores_all, linestyle=all_style['linestyle'],
            marker=all_style['marker'], color=all_style['color'], lw=1.75,
            ms=5.0, label=all_style['label'], zorder=3)
    if scores_cheap is not None:
        ax.plot(budgets_cheap, scores_cheap, linestyle=cheap_style['linestyle'],
                marker=cheap_style['marker'], color=cheap_style['color'], lw=1.75,
                ms=5.0, label=cheap_style['label'], zorder=2)
    if random_data is not None:
        budgets_random, scores_random, p25_random, p75_random = random_scores(
            random_data, metric,
        )
        keep = budgets_random <= max_k
        budgets_random = budgets_random[keep]
        scores_random = scores_random[keep]
        p25_random = p25_random[keep]
        p75_random = p75_random[keep]
        budgets_random = np.concatenate([[0], budgets_random])
        scores_random = np.concatenate([[baseline_score], scores_random])
        p25_random = np.concatenate([[baseline_score], p25_random])
        p75_random = np.concatenate([[baseline_score], p75_random])
        ax.plot(budgets_random, scores_random, linestyle=random_style['linestyle'],
                marker=random_style['marker'], color=random_style['color'], lw=1.55,
                ms=4.5, label=random_style['label'], zorder=1)
        ax.fill_between(budgets_random, p25_random, p75_random,
                        color=random_style['color'], alpha=0.12, linewidth=0, zorder=0)
    ax.scatter([0], [baseline_score], s=64, marker='D', color='white',
               edgecolor='black', linewidth=1.1, zorder=5,
               label='Benchmark median')
    for k in (5, 10):
        ax.scatter([k], [scores_all[k]], s=56, color=all_style['color'],
                   edgecolor='white', linewidth=1.0, zorder=4)
        if scores_cheap is not None:
            ax.scatter([k], [scores_cheap[k]], s=56, color=cheap_style['color'],
                       edgecolor='white', linewidth=1.0, zorder=4)

    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.02),
              fontsize=11.5, frameon=False, borderaxespad=0.0,
              handlelength=1.8, handletextpad=0.45, labelspacing=0.25)
    ax.set_xlabel('Known benchmark count', fontsize=15)
    ax.set_ylabel(YLABEL[metric], fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlim(-0.45, max_k + 0.5)
    ymax = baseline_score
    ymin = float(scores_all[-1])
    if scores_cheap is not None:
        ymin = min(ymin, float(scores_cheap[-1]))
    if random_data is not None and len(scores_random):
        ymax = max(ymax, float(np.nanmax(p75_random)))
        ymin = min(ymin, float(np.nanmin(p25_random)))
    ax.set_ylim(ymin - 0.25, ymax + 0.45)
    ax.grid(axis='y', color=GRAY, alpha=0.18, linewidth=0.7)
    save_fig(out_stem)

    print(f"\n=== Comparison ({metric}) ===")
    print(f"  k= 0  baseline={baseline_score:.3f}")
    for k in (5, 10):
        line = f"  k={k:2d}  all={scores_all[k]:.3f}"
        if scores_cheap is not None:
            line += f"  cost_aware={scores_cheap[k]:.3f}"
        print(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_file', type=str,
                        default='greedy_medape_targets_tall_candidates_tall.json.gz',
                        help='Input JSON filename under results/ (supports .json or .json.gz). '
                             'Falls back to the other extension if default is missing.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=list(SCORE_FIELD),
                        help='Which metric to plot. Defaults to the one used at run time.')
    parser.add_argument('--compare', action='store_true',
                        help='Plot greedy curves, optionally with random and cost-aware baselines.')
    parser.add_argument('--all-in', type=str,
                         default='greedy_medape_targets_tall_candidates_tall.json.gz',
                        help='All-candidates result for --compare.')
    parser.add_argument('--cheap-in', type=str, default=None,
                        help='Optional cost-aware result for --compare.')
    parser.add_argument('--random-in', type=str, default=None,
                        help='Optional Figure-1-style random baseline result for --compare.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output stem under figures/.')
    args = parser.parse_args()

    if args.compare:
        all_name, all_data = load_result(args.all_in)
        cheap_data = None
        cheap_name = None
        if args.cheap_in is not None:
            cheap_name, cheap_data = load_result(args.cheap_in)
        random_data = None
        random_name = None
        if args.random_in is not None:
            random_name, random_data = load_result(args.random_in)
        metric = args.metric or all_data.get('config', {}).get('metric', 'medape')
        assert_greedy_result(all_data, all_name, metric)
        if cheap_data is not None:
            assert_greedy_result(cheap_data, cheap_name, metric)
        if random_data is not None:
            assert_random_result(random_data, random_name)
        plot_compare(
            all_data, cheap_data, metric, args.out or 'bp_probe_evaluation',
            random_data=random_data,
        )
        return

    args.in_file, data = load_result(args.in_file)
    stem = args.out or infer_stem(args.in_file)
    metric = args.metric or data.get('config', {}).get('metric', 'medape')

    trajectory, budgets, scores, added = trajectory_scores(data, metric)
    field = SCORE_FIELD[metric]

    best_idx = int(np.argmin(scores))

    apply_single()
    fig, ax = plt.subplots(figsize=(5.2, 1.9))

    # Main curve (no markers; we scatter them separately with colors)
    ax.plot(budgets, scores, '-', color=VANILLA_BLUE, lw=1.2, zorder=2)

    # Later points in neutral charcoal
    n_label = min(10, len(budgets))
    if len(budgets) > n_label:
        ax.scatter(budgets[n_label:], scores[n_label:], s=35,
                   color=VANILLA_BLUE, zorder=3, edgecolor='white', linewidth=1.0)

    # First n_label points: gradient colors
    point_colors = [PALETTE[i % len(PALETTE)] for i in range(n_label)]
    scatter_handles = []
    for i in range(n_label):
        h = ax.scatter(budgets[i], scores[i], s=55, color=point_colors[i],
                       zorder=4, edgecolor='white', linewidth=1.1,
                       label=f"{i + 1}. {BENCH_NAMES.get(added[i], added[i])}")
        scatter_handles.append(h)

    # Legend inside upper-right whitespace, 2 columns to save vertical space
    ax.legend(
        handles=scatter_handles, loc='upper right',
        fontsize=9, frameon=False, ncol=2,
        handletextpad=0.3, labelspacing=0.22, columnspacing=0.8,
        borderaxespad=0.3,
    )

    # Horizontal reference lines at top-5 and top-10 scores
    for b_ref in (5, 10):
        if b_ref <= len(budgets):
            s_ref = scores[b_ref - 1]
            ax.axhline(s_ref, color=GRAY, ls='--', lw=0.9, alpha=0.5, zorder=1)
            ax.text(
                0.985, s_ref, f"top-{b_ref}: {s_ref:.2f}{VALUE_SUFFIX[metric]} ",
                transform=ax.get_yaxis_transform(), va='bottom', ha='right',
                fontsize=10, color=GRAY,
            )

    ax.set_xlabel('Probe set size', fontsize=11)
    ax.set_ylabel(YLABEL[metric], fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    integer_ticks(ax, 'x')
    ax.set_xlim(0.5, len(budgets) + 0.5)

    save_fig(stem)

    # Print summary
    print(f"\nBest budget: b={budgets[best_idx]} ({BENCH_NAMES.get(added[best_idx], added[best_idx])}), {YLABEL[metric]}={scores[best_idx]:.2f}")
    print(f"\n=== Greedy Trajectory ({metric}) ===")
    for i, t in enumerate(trajectory):
        name = BENCH_NAMES.get(t['added_benchmark_name'], t['added_benchmark_name'])
        marker = ' ◀' if t['step'] == budgets[best_idx] else ''
        print(f"  b={t['step']:2d}  +{name:20s}  {field}={scores[i]:.2f}{VALUE_SUFFIX[metric]}{marker}")


if __name__ == '__main__':
    main()
