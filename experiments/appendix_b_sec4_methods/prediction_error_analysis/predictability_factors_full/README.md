# Benchmark-Side Full Hypothesis Grid

## Paper mapping

- Appendix: `app:bench_analysis`, paragraph `app:predictability_factors_full`.
- Figure: `fig:predictability_factors_full` (`bp_predictability_factors_full.pdf`).
- Source comment: `overleaf/arxiv/appendix.tex`.

## Purpose

This appendix figure expands the main-text benchmark-side prediction-error figure into the full hypothesis grid. It shows all seven active benchmark-side hypotheses against both score-error metrics, using the same canonical Section 4.3 outputs as the main table and figure. The rendered layout splits H1--H3 and H4--H7 into left/right blocks so labels remain readable in the paper PDF.

## Inputs

- `sec4_building_benchpress/error_analysis/benchmark_analysis/H1_*`--`H3_*/results.json`
- `sec4_building_benchpress/error_analysis/benchmark_analysis/H4_*`--`H7_*/ablation_results.json`

## Outputs

- `figures/bp_predictability_factors_full.pdf`
- `figures/bp_predictability_factors_full.png`

## Run

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_b_sec4_methods/prediction_error_analysis/predictability_factors_full
python plot_full.py
```

This script only redraws the appendix figure from existing Section 4.3 result files.
