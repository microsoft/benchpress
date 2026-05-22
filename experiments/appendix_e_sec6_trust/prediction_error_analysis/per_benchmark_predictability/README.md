# Per-Benchmark Predictability

## Paper mapping

- Appendix: `app:reliability_analysis`, paragraph `app:per_benchmark_predictability`.
- Figure: `fig:benchmark_predictability` (`bp_benchmark_predictability.pdf`).
- Source comment: `overleaf/arxiv/appendix.tex`.

## Purpose

This appendix figure reports per-benchmark predictability under the fixed \benchpress{} point predictor. For each model, half of its observed scores are held out and predicted; errors are then aggregated by benchmark column over 10 random seeds.

## Outputs

- `results.json`: per-benchmark error records and aggregate summaries.
- `bp_benchmark_predictability.pdf`: appendix figure copied to Overleaf as `fig:benchmark_predictability`.

## Run

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_e_sec6_trust/prediction_error_analysis/per_benchmark_predictability
python run.py
python plot.py
```

The active directory is `appendix_e_sec6_trust/prediction_error_analysis/per_benchmark_predictability`, matching App E's trust supplement to §6.
