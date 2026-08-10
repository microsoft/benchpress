# §3.3 Interpretation of the top two factors

## Paper mapping

- Section: `sec:bp_svd` (§3.3 Rank-2 Geometry), paragraph "Interpretation of the top two factors".
- Table: `tab:pc_factors` (top and bottom models/benchmarks on PC1 and PC2).

## Purpose

Justify that the leading factor of the score matrix is general capability and that the second factor is not. `pc_loadings.py`:

1. Completes the full matrix with the rank-2 Bias ALS predictor (logit transform, lambda 0.1), column-standardizes, and takes the SVD.
2. Reports, per component, the top/bottom models and benchmarks (`tab:pc_factors`), the benchmark loading distribution (sign agreement, |loading| min/median/max vs uniform, benchmarks needed for half the squared loading), and the Spearman correlation between the PC1 model score and the naive capability ranking (each model's mean z-scored observed score).
3. Runs a completion-free check: per benchmark, the Spearman correlation between its observed scores and the PC1 model score on observed cells only (no completion).

## Inputs

- Score matrix: `benchpress/data/llm_benchmark_data.json` (84 x 133, 2,604 observed cells), via `benchpress.evaluation_harness.M_FULL`.
- Completer: `benchpress.methods.completers.complete_bias_als` (rank-2, lambda 0.1, logit).

## Outputs

- `results.json`: per-component `variance_share`, `models`/`benchmarks` top/bottom (by id), `benchmark_sign_agreement`, `benchmark_abs_loading`, `benchmarks_for_half_squared_loading`, `spearman_with_capability`, and `completion_free_check`. The paper table lists the top/bottom ids with display names from the score matrix.

## Run

Run on GCR CPU (no GPU needed). Not for local Mac.

```bash
cd ~/projects/BenchPress
python experiments/sec3_low_rank/low_rank_structure/factor_interpretation/pc_loadings.py
```

## Reproduction note

Requires the `main`-branch matrix-construction code (its `audit_status` filter yields the 84 x 133 / 2,604-cell matrix) and the full rich `benchpress/data/llm_benchmark_data.json`; an out-of-date build or the `download_data` CSV mirror gives a different matrix and shifts the loadings. No fold cache is needed (the analysis is on the full matrix, not held-out folds).

## Last valid result

Matrix 84 x 133, 2,604 observed cells; env GCR CPU.

- PC1 (variance share 0.779): all 133 benchmarks load with the same sign (agreement 1.0); |loading| min 0.042, median 0.088, max 0.097 against 0.087 uniform; 59 benchmarks reach half the squared loading; Spearman with the naive capability ranking is 0.893.
- PC2 (variance share 0.079): benchmark loadings take both signs (sign agreement 0.53).
- Completion-free check: 131 of 133 benchmarks correlate positively with the PC1 model score on observed cells only (109 above 0.5).

Matches `tab:pc_factors` and the §3.3 interpretation paragraph in `overleaf/arxiv/main_body.tex`.
