# Appendix A.1 inventory tables

## Workflow

```
Need to update Appendix A.1 benchmark/model inventory tables
  ↓
Step 1: Regenerate from the canonical score matrix
  command:
    cd ~/Documents/submission/benchpress/github
    python experiments/appendix_a_sec3_score_matrix/inventory_tables/gen_table.py --write-overleaf
  input:
    benchpress.build_benchmark_matrix.load_score_matrix(return_metadata=True)
  output:
    ~/Documents/submission/benchpress/overleaf/arxiv/appendix.tex
  ↓
Step 2: Check the generated counts
  expected for current matrix:
    84 models, 133 benchmarks, 2,604 observed cells, 23.3% fill
  hard layout rule:
    tables must fit within the page; do not use natural-width l columns for
    free-text fields such as category, benchmark, metric, provider, or model
    name. These fields must use fixed-width p{...} columns, with small
    tabcolsep and scriptsize/tiny where needed.
  ↓
Step 3: Commit/push Overleaf
  command:
    cd ~/Documents/submission/benchpress/overleaf
    git pull --ff-only origin master
    git add arxiv/appendix.tex
    git commit -m "<message>"
    git push
```

## Table source

`gen_table.py` is the source of truth for:
- `tab:benchmarks`, the complete benchmark inventory for the adopted matrix.
- `tab:models`, the complete model inventory for the adopted matrix.

The script reads the canonical filtered matrix instead of hand-maintained table rows. If the score matrix changes, rerun the script rather than editing the table rows by hand.

## Layout guardrails

Never regenerate these tables with unconstrained text columns. The benchmark table is a `longtable`, so it cannot be wrapped in `\resizebox`; all text columns must have explicit `p{...}` widths. The model table is split into two minipages, so each side must use fixed-width `p{...}` columns plus `\tiny` and very small `\tabcolsep`. If a future update changes the matrix or metadata, preserve these width constraints first, then update counts.

Before committing an update, inspect the generated LaTeX and confirm:
- Benchmark table uses `p{...}` for category, benchmark, and metric.
- Model table uses `p{...}` for provider, model, parameter, and release columns.
- No generated text column is plain `l` in the appendix inventory tables.
- Row counts are 133 benchmarks and 84 models for the current matrix.

## Output structure

The generated LaTeX replaces only the Appendix A.1 inventory block between:
- `% ── Benchmark table`
- `\end{table*}` for `tab:models`

The rest of `appendix.tex` is left unchanged.
