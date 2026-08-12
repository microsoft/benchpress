# EEE context comparison

## Purpose

Test whether the larger Every Eval Ever (EEE) matrix gives better predictions
and narrower calibrated intervals than the curated BenchPress matrix on the
same audited target cells.

This is an internal maintenance decision analysis, not a paper-mirrored
experiment. The paper's snapshot and the existing Appendix C portability
experiment remain unchanged.

## Protocol

- BenchPress context: the canonical filtered matrix loaded by
  `benchpress.evaluation_harness`.
- EEE context: percentage-scale columns from the curated EEE `scores.csv`,
  followed by the existing fixed-point filter of at least 15 benchmarks per
  model and 8 models per benchmark.
- Targets: cells whose model and benchmark have an explicit mapping and whose
  score is observed in both matrices. The BenchPress audited score is the
  ground truth.
- Holdout: one shared 3-fold per-model partition with seed 42. Each semantic
  target cell is hidden in both matrices in the same fold.
- Predictor: canonical Logit + rank-2 Bias ALS, lambda 0.1.
- Intervals: the existing matrix-support risk model is cross-fit by fold, then
  calibrated to nominal 90% coverage with leave-fold-out conformal scaling.
- Redundancy ablation: for each fold, remove EEE columns with absolute Pearson
  correlation above 0.98 with a mapped target benchmark, using only the
  fold-training matrix and requiring at least 8 shared models.

The analysis stops before prediction if fewer than 30 matched target cells are
available. This is a sufficiency gate, not a score filter.

## Inputs

- BenchPress canonical JSON selected by the package.
- EEE directory containing `scores.csv` and `scores.meta.json`.
- `mapping.json`, which records manually audited model and benchmark mappings.

## Run

Run on GCR CPU in the `benchpress` environment. Do not run on the local Mac.

```bash
python analysis/eee_context_comparison/run.py \
  --eee-dir ~/eee_matrix \
  --output /tmp/benchpress_eee_context_comparison \
  --phase inventory

python analysis/eee_context_comparison/run.py \
  --eee-dir ~/eee_matrix \
  --output /tmp/benchpress_eee_context_comparison \
  --mapping analysis/eee_context_comparison/mapping.json \
  --phase evaluate
```

## Outputs

- `candidate_report.json`: top model and benchmark mapping candidates.
- `manifest.json`: exact inputs, hashes, matrices, mapping, seed, and protocol.
- `folds/<condition>_fold_<k>.json`: resumable raw predictions, confidence
  features, and redundancy removals for each fold.
- `records.json`: merged per-cell predictions.
- `results.json`: matched accuracy, interval coverage and width, direct
  EEE-to-BenchPress score disagreement, and redundancy-ablation summaries.

If a fold file exists, rerunning validates and reuses it. Delete only the
specific failed fold file before retrying. A changed mapping or input hash
requires a new empty output directory.

## Last valid result

No completed matched-context run yet.
