# App C: BenchPress on other score matrices

## Paper mapping

- Appendix: `app:other_matrices` ("BenchPress on other score matrices").
- Table: `tab:other_matrices`.

## Purpose

Check that BenchPress is not specific to our curated matrix by scoring it on two independently built matrices, curated from the public HELM leaderboard and the community Every Eval Ever (EEE) datastore, under the paper's predictor and protocol.

## How it works

`eval_other_matrices.py` is a thin wrapper over the shared harness, with no duplicated eval logic:

- loads a curated `scores.csv` via `benchpress.data.score_matrix.ScoreMatrix`;
- keeps only percentage-scale benchmark columns (`metric type in {pct}`), because curated matrices mix scales (dollars, Elo, token counts) and a raw MedAE across mixed scales is not meaningful and the logit transform only applies on `[0, 100]`;
- applies the paper's observation filter (>=15 benchmarks per model, >=8 models per benchmark, iterated to a fixed point);
- generates folds with `benchpress.evaluation_harness.holdout_per_model(M=...)` (the matrix-parameterized version) over 10 seeds x 3 folds, the same per-model holdout as the paper;
- predicts with the shared `predict_benchpress_scores` plus per-benchmark-median and global-median baselines;
- aggregates with `compute_prediction_error(..., aggregation="per_group_median")`, the same per-fold-median as the main method comparison.

## Inputs

- HELM matrix: `python -m benchpress.data.helm.curate_matrix --output <dir>` (public HELM Lite/Classic/Capabilities/Long-Context releases).
- EEE matrix: `python -m benchpress.data.eee.curate_matrix --output <dir>` (HF datastore `evaleval/EEE_datastore`; gated, needs an HF token).

## Run

Run on GCR CPU (no GPU needed). Not for local Mac.

```bash
python -m benchpress.data.helm.curate_matrix --output ~/helm_matrix
python -m benchpress.data.eee.curate_matrix  --output ~/eee_matrix   # HF_TOKEN set
python experiments/appendix_c_sec4_methods/other_matrices/eval_other_matrices.py \
    --matrix helm=~/helm_matrix --matrix eee=~/eee_matrix
```

## Last valid result

Percentage-scale, (>=15, >=8)-filtered, per-fold median over 10 seeds x 3 folds; lower is better. Env GCR CPU.

| Predictor | HELM (83 x 207, 9,776) MedAE / MedAPE | EEE (653 x 498, 24,879) MedAE / MedAPE |
|---|---|---|
| BenchPress | 2.87 / 10.67% | 2.89 / 5.02% |
| Per-benchmark median | 4.12 / 15.31% | 7.00 / 11.24% |
| Global median | 19.31 / 46.60% | 18.25 / 23.18% |

BenchPress is the most accurate on both matrices, matching `tab:other_matrices` in `overleaf/arxiv/appendix.tex`.
