# H2 Score Level

## Paper mapping
- Section: `sec:reliability_analysis`, benchmark-side Table `tab:predictability_factors`.
- Parent runner: `../run.sh`.

## Purpose
H2 tests whether benchmark score level predicts BenchPress error. The feature is each benchmark's median observed score in the canonical score matrix.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- Default BenchPress errors from `benchpress/evaluation/default_predictions/benchpress_default/by_benchmark.json`.

## Outputs
- `results.json`: per-benchmark rows with `med_score`, MedAPE, MedAE, and univariate Spearman tests.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/benchmark_analysis
python H2_score_level/analyze.py
```

This is deterministic and sets `SEED=42` through `_shared.py`.
