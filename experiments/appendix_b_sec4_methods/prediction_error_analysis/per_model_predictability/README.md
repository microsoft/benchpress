# Per-Model Predictability

## Paper mapping

- Appendix: `app:model_analysis`, paragraph `app:per_model_predictability`.
- Figure: `fig:model_predictability` (`bp_model_predictability.pdf`).
- Source comment: `overleaf/arxiv/appendix.tex`.

## Purpose

This appendix figure reports per-model predictability under the fixed BenchPress
point predictor. For each model, half of its observed scores are held out and
predicted; errors are then aggregated by **model row** over 10 random seeds.
This is the model-side analog of the per-benchmark predictability appendix in
`appendix_c_sec5_findings/probe_selection/per_benchmark_predictability/`.

## Data reuse (no recompute)

Aggregation only — we reuse the canonical raw predictions stored in
`appendix_c_sec5_findings/probe_selection/per_benchmark_predictability/results.json`
(same predictor, same `holdout_half_per_model`, same 10 seeds). `run.py` loads
that file and re-aggregates by model index `i` instead of benchmark index `j`.

## Outputs

- `results.json`: per-model error records and aggregate summary.
- `bp_model_predictability.pdf`: appendix figure copied to Overleaf as
  `fig:model_predictability`.

## Run

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_b_sec4_methods/prediction_error_analysis/per_model_predictability
python run.py    # aggregates from per_benchmark_predictability raw
python plot.py
```
