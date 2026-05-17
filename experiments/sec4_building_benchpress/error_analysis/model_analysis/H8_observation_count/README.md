# H8 Observation Count

## Paper mapping
- Section: `sec:reliability_analysis`, model-side Table `tab:model_hypotheses` and Figure `fig:error_hypotheses_52`.
- Parent runner: `../run.sh`.

## Purpose
H8 tests whether hiding more observations for a target model increases model-side prediction error. It compares the standard hide-half baseline against 25% and 75% hidden-observation conditions.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- `N_SEEDS = 10` and `SEED = 42` from `../_shared.py`.

## Outputs
- `results.json`:
  - `hide_25pct.tests` and `hide_75pct.tests`: paired model-level Wilcoxon summaries for MedAPE and MedAE.
  - `raw_predictions.baseline`, `raw_predictions.hide_25pct`, and `raw_predictions.hide_75pct`: seeds, model indices, benchmark indices, actuals, and predictions.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/model_analysis
python H8_observation_count/ablation.py
```

This script is deterministic but can be expensive; reuse existing `results.json` unless intentionally rerunning the ablation.
