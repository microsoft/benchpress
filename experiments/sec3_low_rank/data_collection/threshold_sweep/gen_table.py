#!/usr/bin/env python3
"""
Generate `tab:bp_threshold_sweep` rows for §3.1.

Uses the same canonicalization and threshold-filtering code path as the
paper experiments, then prints LaTeX rows showing the resulting matrix
dimensions, observation count, and fill rate.

Usage:
    cd benchpress/github
    python experiments/sec3_low_rank/data_collection/threshold_sweep/gen_table.py
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchpress.build_benchmark_matrix import load_score_matrix


def main() -> None:
    raw, raw_info = load_score_matrix(m_threshold=0, b_threshold=0, deduplicate=False, return_info=True)

    grid_pairs: list[tuple[int, int]] = [
        (10, 8), (10, 12), (10, 16),
        (15, 8), (15, 12), (15, 16),
        (20, 8), (20, 12), (20, 16),
    ]
    chosen = (15, 8)

    rows: list[tuple[int, int, int, int, int, float]] = []
    for tm, tb in grid_pairs:
        _, info = load_score_matrix(m_threshold=tm, b_threshold=tb, deduplicate=True, return_info=True)
        if not info.n_models or not info.n_benchmarks:
            continue
        rows.append((tm, tb, info.n_models, info.n_benchmarks, info.n_observations, info.fill_rate))

    out = []
    out.append("% Auto-generated from experiments/sec3_low_rank/data_collection/threshold_sweep/gen_table.py")
    out.append(r"\begin{tabular}{rr cccc}")
    out.append(r"\toprule")
    out.append(r"$T_M$ & $T_B$ & \#Models & \#Bench. & \#Obs. & Fill \\")
    out.append(r"\midrule")
    out.append(
        rf"\multicolumn{{2}}{{c}}{{(unfiltered)}} & {raw_info.n_models} & {raw_info.n_benchmarks} & "
        f"{raw_info.n_observations:,} & {100*raw_info.fill_rate:.1f}\\% \\\\".replace(",", "{,}")
    )
    out.append(r"\midrule")
    for tm, tb, M, B, O, fill in rows:
        bold = (tm, tb) == chosen
        ob, cb = (r"\textbf{", "}") if bold else ("", "")
        color = r"\cBP" if bold else ""
        obs_str = f"{O:,}".replace(",", "{,}")
        out.append(
            f"{color}{ob}{tm}{cb} & {color}{ob}{tb}{cb} & {color}{ob}{M}{cb} & {color}{ob}{B}{cb} "
            f"& {color}{ob}{obs_str}{cb} & {color}{ob}{100*fill:.1f}\\%{cb} \\\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
