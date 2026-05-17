# BenchPress default predictions

This directory stores the canonical BenchPress default prediction artifact used by downstream analyses.

## Setting

- Predictor: Logit + Bias ALS
- Hyperparameters: rank 2, lambda 0.1
- Evaluation folds: `benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json`
- Matrix: 84 models x 133 benchmarks
- Held-out predictions: 26,040 cells across 10 seeds x 3 folds

## Files

| File | Contents |
|------|----------|
| `predictions.npz` | Raw Section 4.2 fold-level predictions: `fold_id`, `test_i`, `test_j`, `actual`, `predicted`, and `M_pred_by_fold`. |
| `metadata.json` | Predictor setting, fold source, matrix shape, raw array schema, and pooled metrics. |
| `by_benchmark.json` | Per-benchmark metrics plus raw held-out predictions grouped by benchmark. |
| `by_model.json` | Per-model metrics plus raw held-out predictions grouped by model. |

Use these files for paper analyses that need the official BenchPress default prediction errors. Do not read H-specific ablation outputs as the global BenchPress default.
