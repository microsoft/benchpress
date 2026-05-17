#!/usr/bin/env python3
"""Generate complete submatrix SVD table rows across shapes.

Source data: submatrix_sweep.json
Output: printed LaTeX tabular rows (for manual paste into main_body.tex)
"""
import os
from benchpress.io_utils import load_json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    data = load_json(os.path.join(SCRIPT_DIR, 'submatrix_sweep.json'))
    print("% Auto-generated from experiments/sec3_low_rank/low_rank_structure/submatrix_svd/gen_table.py")
    print("% Data: experiments/sec3_low_rank/low_rank_structure/submatrix_svd/submatrix_sweep.json")
    print()
    print(r"\begin{tabular}{ccccc}")
    print(r"\toprule")
    print(r"\textbf{\# Bench.} & \textbf{\# Models} & \cBP\textbf{Stable rank} & \textbf{Var (top 1)} & \textbf{Var (top 2)} \\")
    print(r"\midrule")
    for r in data:
        nb = r['n_benchmarks']
        nm = r['n_models']
        sr = r['stable_rank']
        v1 = r['var_rank1'] * 100
        v2 = r['var_rank2'] * 100
        print(f"{nb} & {nm} & \\cBP {sr:.2f} & {v1:.1f}\\% & {v2:.1f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    main()
