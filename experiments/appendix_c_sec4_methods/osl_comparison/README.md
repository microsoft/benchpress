# Appendix C.4 Comparison to Observational Scaling Laws

## Paper mapping

- Appendix: `app:osl_comparison`.
- Table: `tab:osl_comparison`.

## Purpose

Compare the Observational Scaling Laws (OSL) predictor of Ruan et al. (2024) against BenchPress on the paper's score matrix. OSL fits one regression per benchmark on a densely scored capability block; BenchPress completes arbitrary missing cells.

- `obs_scaling_baseline.py`: run OSL on **our** score matrix and score it against BenchPress on the canonical held-out folds. This produces `tab:osl_comparison`.

The same script contains the OSL reimplementation: `pca_impute` fills the capability block by iterated rank-1 PCA, `fit_capability_pca` takes the top PCs as capability measures, and `SigmoidCapabilityRegression` fits the sigmoid-parametric scaling curve, matching the reference implementation at `github.com/ryoungj/ObsScaling`.

## Inputs

- Score matrix: `benchpress/data/llm_benchmark_data.json` (84 x 133, 2,604 observed cells).
- Folds: `benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json` (10 seeds x 3 folds; loaded via `benchpress.evaluation_harness.load_folds`).
- BenchPress predictor: `benchpress.methods.predictors.predict_benchpress_scores` (rank-2 logit Bias ALS, lambda 0.1).
- Metrics: `benchpress.evaluation_harness.compute_prediction_error` (MedAE / MedAPE only).

## Outputs

- `obs_scaling_baseline.json`: per-fold median MedAE/MedAPE (median over the 10 seeds x 3 folds, matching the main-body method comparison) for OSL (densest-8 block and full-matrix impute) and BenchPress on the canonical folds.

## Run

Run on GCR CPU (no GPU needed). Not for local Mac.

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_c_sec4_methods/osl_comparison
python obs_scaling_baseline.py --workers 8      # our matrix -> tab:osl_comparison
```

`--smoke` runs two folds for a fast sanity check. `obs_scaling_baseline.py` parallelizes folds across `--workers` processes.

## Resume / rerun

The runner writes a single JSON; rerunning overwrites it. Reuse the checked-in JSON for the paper table; do not rerun unless the matrix, folds, the BenchPress predictor, or the OSL reimplementation intentionally change.

## Last valid result

Per-fold median MedAE/MedAPE over the 10 seeds x 3 folds (`aggregation='per_group_median'`), on each method's finite predictions; lower is better:

| Method | MedAPE (%) | MedAE |
|---|---|---|
| OSL, densest-8 block (i) | 9.85 | 5.94 |
| OSL, full-matrix impute (ii) | 21.87 | 12.17 |
| BenchPress | 7.77 | 4.63 |

Matrix 84 x 133, 2,604 observed cells; env GCR CPU. The BenchPress row (4.63 / 7.77%) reproduces the paper's default predictor exactly (`tab:full_grid`), so the appendix table is consistent with the main text. Matches `tab:osl_comparison` in `overleaf/arxiv/appendix.tex`.

## Reproduction note

Reproducing the paper matrix requires all three of: (1) the `main`-branch matrix-construction code, which applies the `audit_status` filter that yields the 84 x 133 / 2,604-cell matrix (an out-of-date build filters differently and gives the wrong shape); (2) the full rich `benchpress/data/llm_benchmark_data.json` (the `download_data` CSV mirror drops audit/cost fields and shifts the numbers); (3) the cached folds file `folds_s10_f3_bs42_ms1.json` (regenerating folds from scratch changes the held-out cells). With all three in place the run gives `benchpress` n=26,040 and MedAE 4.63.
