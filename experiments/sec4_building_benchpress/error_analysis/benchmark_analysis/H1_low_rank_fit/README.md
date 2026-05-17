# H1 Low-Rank Fit

## Paper mapping
- Section: `sec:reliability_analysis`, benchmark-side Table `tab:predictability_factors`.
- Parent runner: `../run.sh`.

## Purpose
H1 tests whether benchmarks with weaker low-rank fit have higher BenchPress prediction error. The feature is the column-wise rank-2 R2 from the canonical filtered score matrix.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- Default BenchPress errors from `benchpress/evaluation/default_predictions/benchpress_default/by_benchmark.json`.

## Outputs
- `results.json`: per-benchmark rows with `rank2_R2`, MedAPE, MedAE, and univariate Spearman tests.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/benchmark_analysis
python H1_low_rank_fit/analyze.py
```

This is deterministic and sets `SEED=42` through `_shared.py`.
