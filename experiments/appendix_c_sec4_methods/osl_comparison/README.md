# Appendix C.4 Comparison to Observational Scaling Laws

## Paper mapping

- Appendix: `app:osl_comparison`.
- Table: `tab:osl_comparison`.
- Backs the qualitative "their setting" claim in the same subsection (weak-to-strong extrapolation).

## Purpose

Compare the Observational Scaling Laws (OSL) predictor of Ruan et al. (2024) against BenchPress. OSL fits one regression per benchmark on a densely scored capability block; BenchPress completes arbitrary missing cells. Two runs answer two questions:

- `obs_scaling_baseline.py`: run OSL on **our** score matrix and score it against BenchPress on the shared held-out cells. This produces `tab:osl_comparison`.
- `obs_scaling_setup.py`: run both predictors on **OSL's own** released benchmarks, splits, and targets, so the comparison does not depend on how our matrix is built. Backs the statement that OSL's scaling formulation is the better-matched tool in a pure weak-to-strong regime.

`observational_scaling.py` is the OSL reimplementation (imported by both runners): `pca_impute` fills the capability block by iterated rank-1 PCA, `fit_capability_pca` takes the top PCs as capability measures, and `SigmoidCapabilityRegression` fits the sigmoid-parametric scaling curve, matching the reference implementation at `github.com/ryoungj/ObsScaling`.

## Inputs

- Score matrix: `benchpress/data/llm_benchmark_data.json` (84 x 133, 2,604 observed cells).
- Folds: `benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json` (10 seeds x 3 folds; loaded via `benchpress.evaluation_harness.load_folds`).
- BenchPress predictor: `benchpress.methods.predictors.predict_benchpress_scores` (rank-2 logit Bias ALS, lambda 0.1).
- Metrics: `benchpress.evaluation_harness.compute_prediction_error` (MedAE / MedAPE only).
- OSL data (`obs_scaling_setup.py` only): CSVs from `github.com/ryoungj/ObsScaling`, downloaded on demand into `osl_data/` (untracked) and pinned by SHA256 in the output JSON. Not vendored.

## Outputs

- `obs_scaling_baseline.json` + `obs_scaling_baseline_raw.npz`: per-fold predictions and pooled MedAE/MedAPE for OSL (densest-8 block and full-matrix impute), BenchPress, and the OSL-imputation and column-median reference rows, on the canonical folds.
- `obs_scaling_setup.json` + `obs_scaling_setup_raw.npz`: per-target and pooled errors on OSL's emergent, agentic, and post-training tasks, with the source-CSV SHA256s.

## Run

Run on GCR CPU (no GPU needed). Not for local Mac.

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_c_sec4_methods/osl_comparison
python obs_scaling_baseline.py --workers 8      # our matrix -> tab:osl_comparison
python obs_scaling_setup.py                     # OSL's own data (downloads osl_data/ on first run)
```

`--smoke` runs two folds / a reduced task set for a fast sanity check. `obs_scaling_baseline.py` parallelizes folds across `--workers` processes.

## Resume / rerun

Each runner writes a single JSON plus a `_raw.npz` of raw per-fold outputs; rerunning overwrites both. Reuse the checked-in JSON for the paper table; do not rerun unless the matrix, folds, the BenchPress predictor, or the OSL reimplementation intentionally change.

## Last valid result

Over the 25,062 held-out cells all three predictors share (canonical folds), lower is better:

| Method | MedAPE (%) | MedAE |
|---|---|---|
| OSL, densest-8 block (i) | 9.89 | 5.92 |
| OSL, full-matrix impute (ii) | 21.61 | 11.99 |
| BenchPress | 7.73 | 4.56 |

Matrix 84 x 133, 2,604 observed cells; env GCR CPU. Matches `tab:osl_comparison` in `overleaf/arxiv/appendix.tex`.
