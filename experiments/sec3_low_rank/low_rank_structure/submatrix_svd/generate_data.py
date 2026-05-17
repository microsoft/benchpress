#!/usr/bin/env python
"""Generate complete-submatrix SVD provenance and table data for §3.3."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchpress.evaluation_harness import M_FULL, MODEL_IDS, BENCH_IDS, OBSERVED
from benchpress.io_utils import write_json

TARGET_BENCHMARK_COUNTS = [4, 7, 10, 13]


def find_largest_complete_submatrix(M, min_benchmarks=5):
    """Find largest complete submatrix by greedy benchmark selection."""
    obs = ~np.isnan(M)
    n_models, n_bench = M.shape
    
    # Sort benchmarks by coverage (descending)
    bench_coverage = obs.sum(axis=0)
    bench_order = np.argsort(-bench_coverage)
    
    best_shape = (0, 0)
    best_rows = []
    best_cols = []
    
    for nb in range(min_benchmarks, n_bench + 1):
        cols = bench_order[:nb]
        rows = np.where(obs[:, cols].all(axis=1))[0]
        if len(rows) > 0 and len(rows) * nb > best_shape[0] * best_shape[1]:
            best_shape = (len(rows), nb)
            best_rows = rows.tolist()
            best_cols = cols.tolist()
    
    return best_rows, best_cols


def find_complete_submatrix_for_benchmark_count(M, n_benchmarks):
    """Select top-coverage benchmarks at a fixed count and keep complete rows."""
    obs = ~np.isnan(M)
    bench_coverage = obs.sum(axis=0)
    bench_order = np.argsort(-bench_coverage)
    cols = bench_order[:n_benchmarks]
    rows = np.where(obs[:, cols].all(axis=1))[0]
    return rows.tolist(), cols.tolist()


def compute_stable_rank(sv):
    """Stable rank = ||A||_F^2 / ||A||_2^2"""
    return np.sum(sv**2) / (sv[0]**2)


def compute_sweep_row(M, n_benchmarks):
    rows_i, cols_i = find_complete_submatrix_for_benchmark_count(M, n_benchmarks=n_benchmarks)
    if len(rows_i) < 2:
        raise ValueError(f"Need at least two complete models for {n_benchmarks} benchmarks; found {len(rows_i)}")
    M_i = M[np.ix_(rows_i, cols_i)]
    M_i_centered = M_i - M_i.mean(axis=0)
    _, s_i, _ = np.linalg.svd(M_i_centered, full_matrices=False)
    total_var = np.sum(s_i**2)
    return {
        "n_benchmarks": len(cols_i),
        "n_models": len(rows_i),
        "stable_rank": compute_stable_rank(s_i),
        "var_rank1": s_i[0]**2 / total_var,
        "var_rank2": (s_i[0]**2 + s_i[1]**2) / total_var if len(s_i) > 1 else s_i[0]**2 / total_var
    }


def main():
    # --- stable_rank_results.json (diagnostic provenance) ---
    print("Computing largest complete submatrix...")
    rows, cols = find_largest_complete_submatrix(M_FULL, min_benchmarks=5)
    M_sub = M_FULL[np.ix_(rows, cols)]
    print(f"  Complete submatrix: {len(rows)} models × {len(cols)} benchmarks")

    # Mean-center columns so SVD = PCA (first component isn't just the mean)
    M_sub_centered = M_sub - M_sub.mean(axis=0)
    U, s, Vt = np.linalg.svd(M_sub_centered, full_matrices=False)
    stable_rank = compute_stable_rank(s)
    cum_var = np.cumsum(s**2) / np.sum(s**2)

    stable_rank_data = {
        "complete_submatrix_shape": [len(rows), len(cols)],
        "model_ids": [MODEL_IDS[i] for i in rows],
        "benchmark_ids": [BENCH_IDS[j] for j in cols],
        "singular_values": s.tolist(),
        "stable_rank": stable_rank,
        "cumulative_variance_explained": cum_var.tolist()
    }

    write_json("stable_rank_results.json", stable_rank_data, indent=2)
    print(f"  Saved stable_rank_results.json (stable rank = {stable_rank:.3f})")

    # --- submatrix_sweep.json (for tab:submatrix) ---
    print("\nComputing submatrix sweep for tab:submatrix...")
    sweep_results = []
    for n_bench in TARGET_BENCHMARK_COUNTS:
        row = compute_sweep_row(M_FULL, n_bench)
        sweep_results.append(row)
        print(f"  {row['n_benchmarks']} bench × {row['n_models']} models: stable_rank={row['stable_rank']:.2f}, var(r1)={row['var_rank1']:.1%}, var(r2)={row['var_rank2']:.1%}")

    write_json("submatrix_sweep.json", sweep_results, indent=2)
    print("  Saved submatrix_sweep.json")


if __name__ == "__main__":
    main()
