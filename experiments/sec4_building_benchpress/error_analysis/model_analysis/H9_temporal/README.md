# H9 Training-Anchor Recency

## Paper mapping
- Section: `sec:reliability_analysis`, model-side Table `tab:model_hypotheses` and Figure `fig:error_hypotheses_52`.
- Parent runner: `../run.sh`.

## Purpose
H9 tests training-anchor recency. Models are split by release date into oldest, middle, and newest thirds. The newest third is always the target set; the experiment compares training on the oldest third against training on the middle third while revealing `k` benchmarks for each target model.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- Release dates from `benchpress.build_benchmark_matrix.MODELS`.
- `N_SEEDS = 10`, `K_VALUES = [1, 3, 5, 8, 10, 15]`.

## Outputs
- `results.json`:
  - group definitions and `k_values`.
  - per-condition aggregate metrics.
  - paired `comparison_A_vs_B` summaries.
  - `raw_predictions_A` and `raw_predictions_B` with seeds, model indices, benchmark indices, actuals, and predictions.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/model_analysis
python H9_temporal/analyze.py
```

The script uses a stable hash-derived condition offset so random streams do not collide across temporal conditions.
