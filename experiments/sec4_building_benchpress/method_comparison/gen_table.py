#!/usr/bin/env python3
"""Generate LaTeX Table 3 (tab:top15) from manifest.json.

This is the Top-15 transform--method configurations table ranked by score-error metric.
"""

import os
from benchpress.io_utils import load_json
from benchpress.table_utils import format_hyperparameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_N = 15

# Transform display names
TRANSFORM_NAMES = {
    'identity': 'Identity',
    'log': 'Log',
    'logit': 'Logit',
    'asinh': 'Arcsinh',
    'sqrt': 'Square root',
    'probit': 'Probit',
    'quantile': 'Quantile',
}

# Method display names
METHOD_NAMES = {
    'Benchmark Mean': 'Bench-Mean',
    'Model Mean': 'Model-Mean',
    'Bench-KNN': 'Bench-KNN',
    'Model-KNN': 'Model-KNN',
    'BenchReg': 'BenchReg',
    'ModelReg': 'ModelReg',
    'Soft-Impute': 'Soft-Impute',
    'Bias ALS': 'Bias ALS',
    'NMF': 'NMF',
    'PMF': 'PMF',
    'Nuclear Norm': 'Nuclear',
    'MLP': 'MLP',
}

def highlight_benchpress(row, *cells):
    """Highlight the selected BenchPress recipe (Logit Bias ALS, lambda=0.1, rank=2)."""
    hp = row.get('hp', {})
    is_selected = (
        row['transform'] == 'logit'
        and row['method'] == 'Bias ALS'
        and hp.get('lam') == 0.1
        and hp.get('rank') == 2
    )
    if not is_selected:
        return cells
    return tuple(rf"\cBP\textbf{{{cell}}}" for cell in cells)

def load_rows():
    """Load all completed transform--method--HP configurations from manifest."""
    manifest = load_json(os.path.join(SCRIPT_DIR, "manifest.json"))

    rows = []
    for row in manifest['completed']:
        rows.append({
            'transform': row['transform'],
            'method': row['method'],
            'medape': row['medape_median'],
            'medae': row['medae_median'],
            'coverage': row.get('coverage', 1.0),
            'hp': row.get('hp', {}),
        })
    return rows

def gen_table():
    rows = load_rows()
    
    # Sort independently by the two score-error metrics used in Section 4.
    top_medape = sorted(rows, key=lambda x: x['medape'])[:TOP_N]
    top_medae = sorted(rows, key=lambda x: x['medae'])[:TOP_N]
    
    lines = []
    lines.append(r"\begin{tabular}{@{}rlllr@{\hspace{12pt}}rlllr@{}}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{5}{c}{\textbf{MedAPE (\%) $\downarrow$}} &")
    lines.append(r"\multicolumn{5}{c}{\textbf{MedAE $\downarrow$}} \\")
    lines.append(r"\cmidrule(r){1-5} \cmidrule(l){6-10}")
    lines.append(r"\# & Transform & Method & Hyperparameter & Value &")
    lines.append(r"\# & Transform & Method & Hyperparameter & Value \\")
    lines.append(r"\midrule")
    
    for i in range(TOP_N):
        m1, m2 = top_medape[i], top_medae[i]
        
        t1 = TRANSFORM_NAMES.get(m1['transform'], m1['transform'])
        t2 = TRANSFORM_NAMES.get(m2['transform'], m2['transform'])
        
        n1 = METHOD_NAMES.get(m1['method'], m1['method'])
        n2 = METHOD_NAMES.get(m2['method'], m2['method'])
        
        hp1 = format_hyperparameters(m1['hp'])
        hp2 = format_hyperparameters(m2['hp'])
        
        c1 = f"({m1['coverage']:.0%})".replace('%', r'\%')
        c2 = f"({m2['coverage']:.0%})".replace('%', r'\%')
        
        v1 = f"{m1['medape']:.1f} {c1}".strip()
        v2 = f"{m2['medae']:.2f} {c2}".strip()
        t1, n1, hp1, v1 = highlight_benchpress(m1, t1, n1, hp1, v1)
        t2, n2, hp2, v2 = highlight_benchpress(m2, t2, n2, hp2, v2)
        
        line = f"{i+1} & {t1} & {n1} & {hp1} & {v1} & {i+1} & {t2} & {n2} & {hp2} & {v2} \\\\"
        lines.append(line)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    
    return "\n".join(lines)

def gen_full_table():
    """DEPRECATED: moved to experiments/appendix_b_sec4_methods/method_comparison/gen_full_table.py
    (Appendix B owns the full 84-row transform-method table; sec4 only owns the top-10 leaderboard.)
    """
    raise RuntimeError(
        "Full table generation moved to "
        "experiments/appendix_b_sec4_methods/method_comparison/gen_full_table.py — run that instead."
    )


if __name__ == "__main__":
    print(gen_table())
