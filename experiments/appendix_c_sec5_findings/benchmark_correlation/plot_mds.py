#!/usr/bin/env python3
"""MDS view of benchmark similarity from pairwise correlations."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.all_methods import (  # noqa: E402
    M_FULL, N_BENCH, BENCH_IDS, BENCH_NAMES, BENCH_CATS,
    _is_pct_bench, _to_logit,
)
from benchpress.evaluation_harness import col_normalize  # noqa: E402
from benchpress.plot_helpers.visual_identity import (  # noqa: E402
    CAT_COLORS, CHARCOAL, GRAY, MEMENTO_MAGENTA, VANILLA_BLUE,
    ANSWER_VIOLET, CYAN_TEAL, SOL_BASE2, DOUBLE_COL,
)
from benchpress.io_utils import load_json  # noqa: E402

MIN_SHARED = 5
HIGHLIGHTS = {
    'math_500': 'MATH-500',
    'aime_2025': 'AIME 2025',
    'arc_agi_1': 'ARC-AGI-1',
    'arc_agi_2': 'ARC-AGI-2',
}
GREEDY_RESULTS = {
    'medape': (
        'MedAPE greedy probes',
        os.path.join(
            REPO_ROOT,
            'experiments/sec5_findings/optimal_probe/results/'
            'greedy_medape_targets_tall_candidates_tall.json.gz',
        ),
        'bp_benchmark_correlation_mds_medape_greedy',
    ),
    'medae': (
        'MedAE greedy probes',
        os.path.join(
            REPO_ROOT,
            'experiments/sec5_findings/optimal_probe/results/'
            'greedy_medae_targets_tall_candidates_tall.json.gz',
        ),
        'bp_benchmark_correlation_mds_medae_greedy',
    ),
}
SHORT_NAMES = {
    'gpqa_diamond': 'GPQA',
    'hle': 'HLE',
    'aime_2024': 'AIME 2024',
    'aime_2025': 'AIME 2025',
    'mmlu_pro': 'MMLU-Pro',
    'arc_agi_1': 'ARC-AGI-1',
    'arc_agi_2': 'ARC-AGI-2',
    'aider_polyglot_diff': 'Aider',
    'livecodebench': 'LiveCodeBench',
    'terminal_bench': 'Terminal-Bench',
    'swe_bench_verified': 'SWE-bench',
    'codeforces_rating': 'Codeforces',
    'math_500': 'MATH-500',
    'hle_text': 'HLE Text',
    'gdpval_aa_elo': 'GDPval',
}


def _logit_z_matrix():
    is_pct = np.array([_is_pct_bench(j, M_FULL) for j in range(N_BENCH)])
    m_logit = M_FULL.copy()
    for j in range(N_BENCH):
        if is_pct[j]:
            valid = np.isfinite(M_FULL[:, j])
            m_logit[valid, j] = _to_logit(M_FULL[valid, j])
    m_z, _, _ = col_normalize(m_logit)
    return m_z


def _pairwise_abs_corr(m_z):
    corr = np.eye(N_BENCH)
    counts = np.zeros((N_BENCH, N_BENCH), dtype=int)
    for a in range(N_BENCH):
        for b in range(a + 1, N_BENCH):
            mask = np.isfinite(M_FULL[:, a]) & np.isfinite(M_FULL[:, b])
            n = int(mask.sum())
            counts[a, b] = counts[b, a] = n
            if n < MIN_SHARED:
                corr[a, b] = corr[b, a] = np.nan
                continue
            x = m_z[mask, a]
            y = m_z[mask, b]
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                corr[a, b] = corr[b, a] = np.nan
                continue
            if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
                corr[a, b] = corr[b, a] = np.nan
                continue
            r = float(sp_stats.pearsonr(x, y).statistic)
            corr[a, b] = corr[b, a] = abs(r) if np.isfinite(r) else np.nan
    return corr, counts


def _classical_mds(distance):
    n = distance.shape[0]
    d2 = distance ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals = np.maximum(vals[:2], 0)
    return vecs[:, :2] * np.sqrt(vals)


def _annotate(ax, xy, text, offset):
    ax.annotate(
        text, xy=xy, xytext=offset, textcoords='offset points',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=CHARCOAL,
        arrowprops=dict(arrowstyle='-', color=CHARCOAL, lw=0.9),
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=SOL_BASE2, lw=0.8, alpha=0.94),
        zorder=5,
    )


def _setup_background(ax, coords, excluded=()):
    excluded = set(excluded)
    for cat in sorted(set(BENCH_CATS)):
        idx = [i for i, c in enumerate(BENCH_CATS) if c == cat and BENCH_IDS[i] not in excluded]
        if not idx:
            continue
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            s=28, color=CAT_COLORS.get(cat, GRAY), alpha=0.26,
            edgecolors='none', label=cat, zorder=1,
        )


def _format_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel('MDS dimension 1')
    ax.set_ylabel('MDS dimension 2')
    ax.grid(True, color=SOL_BASE2, lw=0.8, alpha=0.8)
    ax.set_aspect('equal', adjustable='datalim')
    for spine in ax.spines.values():
        spine.set_color(SOL_BASE2)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        frameon=False, ncol=1, title='Benchmark type',
        fontsize=9, title_fontsize=10,
    )


def _save(fig, stem):
    out_base = os.path.join(SCRIPT_DIR, 'figures', stem)
    fig.tight_layout()
    fig.savefig(out_base + '.pdf')
    fig.savefig(out_base + '.png')
    plt.close(fig)
    print(f"Saved {out_base}.pdf")
    print(f"Saved {out_base}.png")


def _plot_focus_pairs(coords, corr, counts):
    id_to_idx = {bid: i for i, bid in enumerate(BENCH_IDS)}
    with plt.rc_context(DOUBLE_COL):
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        _setup_background(ax, coords, HIGHLIGHTS)
        for cat in sorted(set(BENCH_CATS)):
            pass

        highlight_colors = {
            'math_500': MEMENTO_MAGENTA,
            'aime_2025': VANILLA_BLUE,
            'arc_agi_1': ANSWER_VIOLET,
            'arc_agi_2': CYAN_TEAL,
        }
        for bid, label in HIGHLIGHTS.items():
            i = id_to_idx[bid]
            ax.scatter(
                coords[i, 0], coords[i, 1],
                s=150, marker='*', color=highlight_colors[bid],
                edgecolor=CHARCOAL, linewidth=0.8, zorder=4,
            )

        offsets = {
            'math_500': (-46, 24),
            'aime_2025': (48, 18),
            'arc_agi_1': (-48, -26),
            'arc_agi_2': (50, -28),
        }
        for bid, label in HIGHLIGHTS.items():
            i = id_to_idx[bid]
            _annotate(ax, coords[i], label, offsets[bid])

        def _connect(a, b, label, text_offset):
            ia, ib = id_to_idx[a], id_to_idx[b]
            ax.plot(
                [coords[ia, 0], coords[ib, 0]],
                [coords[ia, 1], coords[ib, 1]],
                color=CHARCOAL, lw=1.0, ls='--', alpha=0.55, zorder=2,
            )
            mid = (coords[ia] + coords[ib]) / 2
            ax.annotate(
                label, xy=mid, xytext=text_offset, textcoords='offset points',
                ha='center', va='center', fontsize=10, color=CHARCOAL,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85),
                zorder=6,
            )

        math_i, aime_i = id_to_idx['math_500'], id_to_idx['aime_2025']
        arc1_i, arc2_i = id_to_idx['arc_agi_1'], id_to_idx['arc_agi_2']
        _connect(
            'math_500', 'aime_2025',
            f"|r|={corr[math_i, aime_i]:.2f}, n={counts[math_i, aime_i]}",
            (0, 22),
        )
        _connect(
            'arc_agi_1', 'arc_agi_2',
            f"|r|={corr[arc1_i, arc2_i]:.2f}, n={counts[arc1_i, arc2_i]}",
            (0, -22),
            )

        _format_axes(ax, 'Benchmark similarity from pairwise correlations')
        _save(fig, 'bp_benchmark_correlation_mds')
        print(f"MATH-500/AIME 2025: |r|={corr[math_i, aime_i]:.3f}, n={counts[math_i, aime_i]}")
        print(f"ARC-AGI-1/2: |r|={corr[arc1_i, arc2_i]:.3f}, n={counts[arc1_i, arc2_i]}")


def _load_greedy_sequence(path):
    data = load_json(path)
    return [step['added_benchmark'] for step in data['trajectory'][:10]]


def _label_offset(coord):
    norm = np.linalg.norm(coord)
    if norm < 1e-9:
        return (14, 14)
    direction = coord / norm
    return (float(direction[0] * 28), float(direction[1] * 28))


def _plot_greedy(coords, title, result_path, stem):
    selected = _load_greedy_sequence(result_path)
    id_to_idx = {bid: i for i, bid in enumerate(BENCH_IDS)}
    with plt.rc_context(DOUBLE_COL):
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        _setup_background(ax, coords, selected)
        for rank, bid in enumerate(selected, start=1):
            i = id_to_idx[bid]
            if rank <= 5:
                marker = '*'
                size = 175
                color = MEMENTO_MAGENTA
                label = 'Top 1-5' if rank == 1 else None
            else:
                marker = 's'
                size = 95
                color = VANILLA_BLUE
                label = 'Top 6-10' if rank == 6 else None
            ax.scatter(
                coords[i, 0], coords[i, 1],
                s=size, marker=marker, color=color,
                edgecolor=CHARCOAL, linewidth=0.75, zorder=4, label=label,
            )
            offset = _label_offset(coords[i])
            ax.annotate(
                f"{rank}. {SHORT_NAMES.get(bid, BENCH_NAMES.get(bid, bid))}",
                xy=coords[i], xytext=offset, textcoords='offset points',
                ha='center', va='center', fontsize=9.5, color=CHARCOAL,
                arrowprops=dict(arrowstyle='-', color=CHARCOAL, lw=0.65),
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=SOL_BASE2, lw=0.7, alpha=0.92),
                zorder=6,
            )
        _format_axes(ax, title)
        _save(fig, stem)
        print(title + ': ' + ', '.join(selected))


def run():
    os.makedirs(os.path.join(SCRIPT_DIR, 'figures'), exist_ok=True)
    m_z = _logit_z_matrix()
    corr, counts = _pairwise_abs_corr(m_z)
    finite = corr[np.isfinite(corr) & ~np.eye(N_BENCH, dtype=bool)]
    fill_value = float(np.nanmedian(finite))
    corr_filled = np.where(np.isfinite(corr), corr, fill_value)
    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr_filled)))
    np.fill_diagonal(distance, 0.0)
    coords = _classical_mds(distance)

    _plot_focus_pairs(coords, corr, counts)
    for _, (title, result_path, stem) in GREEDY_RESULTS.items():
        _plot_greedy(coords, title, result_path, stem)


if __name__ == '__main__':
    run()
