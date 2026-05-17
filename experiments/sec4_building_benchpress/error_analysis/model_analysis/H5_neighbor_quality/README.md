# H5 Strong-Peer Presence

## Paper mapping
- Section: `sec:reliability_analysis`, model-side Table `tab:model_hypotheses` and Figure `fig:error_hypotheses_52`.
- Parent runner: `../run.sh`.

## Purpose
H5 tests whether removing highly correlated peer models hurts prediction for a target model. Peers are defined by pairwise absolute Pearson correlation at `NEIGHBOR_THRESH = 0.95`.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- Per-model hide-half baseline from `../_shared.py` with `N_SEEDS = 10` and `SEED = 42`.

## Outputs
- `results.json`:
  - `tests`: paired model-level Wilcoxon summaries for MedAPE and MedAE.
  - `raw_predictions.baseline` and `raw_predictions.ablation`: seeds, model indices, benchmark indices, actuals, and predictions.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/model_analysis
python H5_neighbor_quality/ablation.py
```

This script is deterministic but can be expensive; reuse existing `results.json` unless intentionally rerunning the ablation.
