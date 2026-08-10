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

- `obs_scaling_baseline.json`: pooled MedAE/MedAPE for OSL (densest-8 block and full-matrix impute) and BenchPress on the canonical folds.

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

Metrics are from `benchpress.evaluation_harness.compute_prediction_error` on each method's finite predictions over the canonical held-out folds; lower is better:

| Method | MedAPE (%) | MedAE |
|---|---|---|
| OSL, densest-8 block (i) | 9.90 | 5.92 |
| OSL, full-matrix impute (ii) | 21.61 | 11.99 |
| BenchPress | 7.84 | 4.59 |

Matrix 84 x 133, 2,604 observed cells; env GCR CPU. Matches `tab:osl_comparison` in `overleaf/arxiv/appendix.tex`.
