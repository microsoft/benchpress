#!/usr/bin/env python3
"""Appendix B — full 84-row transform x method leaderboard (tab:full_grid).

Reads the same results.json produced by sec4 method_comparison/run.py.
"""
import os
from benchpress.io_utils import load_json
from benchpress.table_utils import format_hyperparameters

SEC4_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'sec4_building_benchpress', 'method_comparison',
)
RESULTS_PATH = os.path.normpath(os.path.join(SEC4_DIR, 'results.json'))

TRANSFORM_NAMES = {
    'identity': 'Identity', 'log': 'Log', 'logit': 'Logit', 'asinh': 'Arcsinh',
    'sqrt': 'Square root', 'probit': 'Probit', 'quantile': 'Quantile',
}
METHOD_NAMES = {
    'Benchmark Mean': 'Bench-Mean', 'Model Mean': 'Model-Mean',
    'Bench-KNN': 'Bench-KNN', 'Model-KNN': 'Model-KNN',
    'BenchReg': 'BenchReg', 'ModelReg': 'ModelReg',
    'Soft-Impute': 'Soft-Impute', 'Bias ALS': 'Bias ALS',
    'NMF': 'NMF', 'PMF': 'PMF', 'Nuclear Norm': 'Nuclear', 'MLP': 'MLP',
}

def gen_full_table():
    results = load_json(RESULTS_PATH)
    rows = []
    for tname, methods in results.items():
        for mname, r in methods.items():
            rows.append({
                'transform': tname, 'method': mname,
                'medape': r['medape_median'], 'medae': r['medae_median'],
                'coverage': r.get('coverage', 1.0),
                'hp': r.get('best_hp', {}),
            })
    rows.sort(key=lambda x: x['medape'])

    L = []
    L.append(r"\begin{longtable}{@{}rlllrrr@{}}")
    L.append(r"\caption{Full transform $\times$ method grid: all 84 configurations from \Cref{sec:method_comparison}, sorted by $\mathsf{MedAPE}$. Each row reports the best hyperparameter setting for that transform--method pair, evaluated as the median over 10 seeds $\times$ 3 folds in standardized space.}\label{tab:full_grid} \\")
    L.append(r"\toprule")
    L.append(r"\# & Transform & Method & Hyperparameter & MedAPE (\%) $\downarrow$ & MedAE $\downarrow$ & Cov. \\")
    L.append(r"\midrule")
    L.append(r"\endfirsthead")
    L.append(r"\multicolumn{7}{c}{\tablename\ \thetable\ -- continued from previous page} \\")
    L.append(r"\toprule")
    L.append(r"\# & Transform & Method & Hyperparameter & MedAPE (\%) $\downarrow$ & MedAE $\downarrow$ & Cov. \\")
    L.append(r"\midrule")
    L.append(r"\endhead")
    L.append(r"\midrule \multicolumn{7}{r}{\textit{continued on next page}} \\")
    L.append(r"\endfoot")
    L.append(r"\bottomrule")
    L.append(r"\endlastfoot")

    for i, row in enumerate(rows, 1):
        t = TRANSFORM_NAMES.get(row['transform'], row['transform'])
        n = METHOD_NAMES.get(row['method'], row['method'])
        hp = format_hyperparameters(row['hp'])
        cov = f"{row['coverage']*100:.0f}\\%"
        L.append(f"{i} & {t} & {n} & {hp} & {row['medape']:.2f} & {row['medae']:.2f} & {cov} \\\\")

    L.append(r"\end{longtable}")
    return "\n".join(L)


if __name__ == "__main__":
    print(gen_full_table())
