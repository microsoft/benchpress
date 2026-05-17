# Section 4.4 Confidence Calibration

## Paper mapping

- Main text: `\Cref{sec:confidence_calibration}`
- Purpose: evaluate whether BenchPress can tell when its own score predictions are reliable enough for benchmark triage.

## Purpose

This experiment keeps the point predictor fixed to the current BenchPress recipe: Logit Bias ALS with `rank=2` and `lam=0.1`. It then attaches uncertainty scores or intervals to the same held-out predictions used in Section 4.2. The key question is whether high-confidence predictions have lower held-out error than low-confidence predictions, and whether prediction intervals are calibrated.

## Confidence generators

The confidence methods are implemented in `benchpress/methods/confidence.py`. This experiment imports those package methods and evaluates them on the paper's held-out folds; it should not redefine the confidence logic locally.

All three main generators train a leave-fold-out multilayer perceptron (MLP) to predict `log1p(abs(predicted - actual))` for the fixed Logit Bias ALS point predictor. At test time, the MLP sees only pre-evaluation features for the hidden cell and outputs an uncertainty/risk score; larger means less confident.

1. **Ensemble-spread uncertainty model**: uses only prediction-spread features. For each held-out cell, it compares the selected Logit Bias ALS prediction against two prediction stacks:
   - Same-family Logit Bias ALS variants: rank 2 with `lam` in `{0.01, 0.1, 1.0}`. The `lam=0.1` member is the selected point predictor; the other two measure nearby regularization sensitivity under the same transform and method.
   - Strong full-coverage alternatives: sort Section 4.2 transform-method rows by median MedAPE, require `coverage >= 0.999`, exclude Logit Bias ALS rows already represented in the same-family stack, and take the first 12. The checked-in `confidence_scores.npz` metadata is the source of truth for the exact files: Probit Bias ALS (`lam=0.1`), Quantile Bias ALS (`lam=0.1`), Identity Bias ALS (`lam=0.1`), Quantile Soft-Impute (`rank=2`), Logit Soft-Impute (`rank=2`), Probit Soft-Impute (`rank=2`), Arcsinh Bias ALS (`lam=0.1`), Square-root Bias ALS (`lam=0.1`), Identity Soft-Impute (`rank=2`), Logit Model-KNN (`k=10`), Probit Model-KNN (`k=10`), and Identity Model-KNN (`k=10`).

   For each stack, the feature vector contains standard deviation, median absolute deviation, central 80% span (`p90 - p10`), and distance from the selected Logit Bias ALS prediction to the stack median. All nonnegative features are transformed with `log1p` before split-local standardization.
2. **Matrix-support uncertainty model**: uses only training-matrix evidence features. The feature vector contains target-model observation count, target-benchmark observation count, target-model median score, target-benchmark median score, target-benchmark score dispersion, strongest peer-model correlation and overlap, and strongest benchmark-neighbor correlation. The peer model and benchmark neighbor are chosen by absolute correlation in the training matrix. The benchmark-neighbor overlap feature is excluded because the stricter H7 ablation does not support it as a joint benchmark-side factor.
3. **Hybrid uncertainty model**: uses both the ensemble-spread features and the matrix-support features in one MLP.
4. **Diagnostic generators**: raw Bias ALS hyperparameter disagreement and raw strong-method disagreement are retained in the cache/results for ablations and debugging, but they are not part of the main figure.
5. **Leave-fold-out conformal scaling**: post-processes any raw uncertainty score into a calibrated 90% interval by fitting the scale multiplier on all other folds and evaluating on the held-out fold. This is a calibration wrapper, not a standalone generator.

All three MLP generators standardize inputs within each training split and use ReLU, Adam, `alpha=1e-3`, `learning_rate_init=3e-3`, early stopping, and `max_iter=500`. Hidden layers are selected inside the training folds from `(16,)`, `(32,)`, and `(64, 32)`. The risk score for each evaluated fold is always produced by an MLP trained on the other folds only.

## How to run

This experiment trains leave-fold-out MLP risk models and is slow in serial. Recommended: run the 12 fold shards in parallel on a CPU machine with at least 24 cores, then merge into the final `confidence_scores.npz` / `results.json` pair.

```bash
# Run each fold shard (run all 12 in parallel, e.g. via xargs / a job scheduler)
for i in $(seq 0 11); do
  python experiments/sec4_building_benchpress/confidence_calibration/run.py \
    --num-fold-shards 12 --fold-shard-index $i \
    --scores-path shards/shard_$i.npz --skip-results &
done
wait

# Merge shards into the final artifacts
python experiments/sec4_building_benchpress/confidence_calibration/run.py \
  --merge-scores --scores-path shards/shard_*.npz
```

## Parallel execution

Unit of work: one fold shard. Each shard writes a `shards/shard_*.npz` file. The merge step requires all fold shards and fails if any uncertainty array still has missing values.

After the job finishes, copy the four final artifacts back to this directory:

- `confidence_scores.npz`
- `results.json`
- `bp_confidence_calibration.pdf`
- `bp_confidence_calibration.png`

If running on a remote machine, copy the four final artifacts back to this directory once the job finishes:

For debugging only, `run.py` can still be called directly. Use `--num-fold-shards`, `--fold-shard-index`, `--scores-path`, and `--skip-results` to produce shard `.npz` files, then `--merge-scores` to combine them into final `confidence_scores.npz` and `results.json`.

## Inputs

- `experiments/sec4_building_benchpress/method_comparison/results.json`
- `experiments/sec4_building_benchpress/method_comparison/manifest.json`
- The Section 4.2 prediction shards for the target Logit Bias ALS ensemble and the top full-coverage comparison methods (`logit_bias_als_ensemble_*`).
- `benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json`

## Outputs

- `confidence_scores.npz`: per-cell source-of-truth cache for confidence evaluation.
- `results.json`: aggregate confidence metrics and risk-coverage curves.
- `figures/bp_confidence_calibration.{pdf,png}`: compact plot for the main text or appendix.

`confidence_scores.npz` contains:

| Key | Shape / type | Meaning |
|-----|--------------|---------|
| `fold_id` | `(n_test,)` int | Fold index for each held-out cell |
| `test_i`, `test_j` | `(n_test,)` int | Held-out model and benchmark indices |
| `actual` | `(n_test,)` float | True held-out score |
| `predicted` | `(n_test,)` float | BenchPress point prediction |
| `bias_als_hp_disagreement_uncertainty` | `(n_test,)` float | Robust spread across same-method HP variants |
| `strong_method_disagreement_uncertainty` | `(n_test,)` float | Robust spread across strong complete predictors |
| `disagreement_uncertainty` | `(n_test,)` float | Cross-fit MLP risk score trained on `log1p(abs(error))` from disagreement features only |
| `structural_support_uncertainty` | `(n_test,)` float | Cross-fit MLP risk score trained on `log1p(abs(error))` from Section 4.3-style structural features only |
| `combined_risk_model_uncertainty` | `(n_test,)` float | Cross-fit MLP risk score trained on `log1p(abs(error))` from structural plus disagreement features |
| `<method>_uncertainty` | `(n_test,)` float | Optional additional generator; larger means less confident |
| `<method>_lower`, `<method>_upper` | `(n_test,)` float | Optional nominal interval endpoints |
| `metadata_json` | scalar string | Input files, method settings, and matrix shape |

## Metrics

The point predictor is fixed throughout this experiment. Metric differences come from ranking or intervaling the same Logit Bias ALS predictions, not from changing the score predictor.

- **Confidence-error ranking**: Spearman rank correlation between predicted risk and realized absolute error. This asks whether the confidence layer orders cells correctly.
- **Selective MedAPE**: sort cells by predicted risk, keep the most trusted 100%, 80%, 60%, 40%, or 20%, and compute MedAPE/MedAE of the fixed point predictions on the retained subset. This asks which predictions can be trusted for evaluation triage.
- **Conformal interval width**: leave-fold-out conformal scaling turns each raw risk score into a 90% interval. Coverage checks calibration; median interval width compares sharpness among methods with similar coverage.
- **Uncertainty terciles**: MedAPE in low-, medium-, and high-uncertainty cells for a coarse monotonicity check.

## Resume / rerun

Reuse the checked-in artifacts by default. Do not rerun this experiment merely to support plotting, paper edits, package cleanup, or `predict.py` work.

Only rerun if the score matrix, canonical folds, Section 4.2 method-comparison prediction shards, or confidence method definitions intentionally change. For a failed or interrupted rerun, keep any completed `shards/shard_*.npz` files and rerun only the missing fold shards with `--fold-shard-index`; then merge all shard files with `--merge-scores`.

## Existing reusable artifacts

These checked-in artifacts are the canonical §4.4 result set and should be reused unless one of the rerun conditions above is met:

- `confidence_scores.npz`: source-of-truth per-cell confidence cache; last touched in commit `4ea402b`.
- `results.json`: aggregate metrics used by the paper; last touched in commit `6fb7ce6`.
- `figures/bp_confidence_calibration.pdf` and `.png`: rendered paper figure; latest PDF last touched in commit `15aa89f`.

The canonical result uses the current 84 x 133 matrix, the cached Section 4.2 folds, and BenchPress = Logit Bias ALS with `rank=2`, `lam=0.1`. It evaluates 26,040 held-out predictions. Point prediction performance is MedAPE 7.76 and MedAE 4.60.

| Method | Spearman risk vs. abs error | Top-20% MedAPE | 90% interval width | Coverage |
|--------|-----------------------------|----------------|--------------------|----------|
| Ensemble-spread uncertainty model | 0.495 | 3.29 | 27.65 | 0.900 |
| Matrix-support uncertainty model | 0.475 | 2.87 | 29.27 | 0.900 |
| Hybrid uncertainty model | 0.536 | 2.71 | 27.17 | 0.899 |

## Reuse boundary for `predict.py`

The §4.4 artifacts are evaluation artifacts: they contain cross-fit held-out uncertainty scores and aggregate paper metrics. They should be reused for paper figures, tables, text, and confidence-method comparisons. A deploy-time `predict.py` calibrator is a separate artifact because it must store a fitted scaler/model plus conformal scale for future missing cells; do not train or regenerate that deploy artifact until the desired deployment protocol is explicitly specified.
