# H7 Same-Provider Evidence

## Paper mapping
- Section: `sec:reliability_analysis`, model-side Table `tab:model_hypotheses`.
- Parent runner: `../run.sh`.

## Purpose
H7 tests whether same-provider peer evidence affects model-side prediction error. It compares a per-model hide-half baseline against an ablation that removes same-provider peer information.

## Inputs
- Canonical score matrix and provider metadata from `benchpress.all_methods`.
- Per-model hide-half baseline from `../_shared.py` with `N_SEEDS = 10` and `SEED = 42`.

## Outputs
- `results.json`:
  - `tests`: paired model-level Wilcoxon summaries for MedAPE and MedAE.
  - `raw_predictions.baseline` and `raw_predictions.ablation`: seeds, model indices, benchmark indices, actuals, and predictions.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/model_analysis
python H7_family_peers/ablation.py
```

This script is deterministic but can be expensive; reuse existing `results.json` unless intentionally rerunning the ablation.
