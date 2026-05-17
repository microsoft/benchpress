# H3 Score Dispersion

## Paper mapping
- Section: `sec:reliability_analysis`, benchmark-side Table `tab:predictability_factors` and Figure `fig:predictability_factors_51`.
- Parent runner: `../run.sh`.

## Purpose
H3 tests whether benchmarks with wider observed-score spread have different prediction error. The feature is the sample standard deviation of observed scores for each benchmark.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- Default BenchPress errors from `benchpress/evaluation/default_predictions/benchpress_default/by_benchmark.json`.

## Outputs
- `results.json`: per-benchmark rows with `std_score`, MedAPE, MedAE, and univariate Spearman tests.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/benchmark_analysis
python H3_score_spread/analyze.py
```

This is deterministic and sets `SEED=42` through `_shared.py`.
