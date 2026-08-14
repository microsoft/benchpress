# Appendix C.2 Full Method Comparison Table

## Paper mapping

- Appendix: `app:method_comparison`.
- Tables: `tab:full_grid` (`gen_full_table.py`), `tab:model_selection` (`gen_selection_table.py`).

## Purpose

This directory generates the validation-error and outer-test-error leaderboards for Appendix C.2. It is a read-only derivative of the main §4.2 method comparison run.

## Inputs

- `../../sec4_building_benchpress/method_comparison/manifest.json`: the canonical 329-configuration sweep used by the main-text leaderboard; `gen_selection_table.py` requires the validation shards to match it exactly, then retains the 203 configurations with 100% outer-test coverage.
- `../../sec4_building_benchpress/method_comparison/inner_scores/*.npz`: per-configuration inner-validation metrics used by `gen_selection_table.py`.
- `../../sec4_building_benchpress/method_comparison/results_nested.json`: per-pair outer-test leaderboard with hyperparameters selected on nested validation cells, used by `gen_full_table.py`.

## Outputs

- stdout: LaTeX `longtable` for `tab:full_grid`, or the Top-15 validation leaderboard for `tab:model_selection`.

## Run

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_c_sec4_methods/method_comparison
python gen_full_table.py
```

or:

```bash
bash run.sh
```

## Resume / rerun

No experiment runs here. Regenerate `tab:model_selection` after `inner_scores/*.npz` or `manifest.json` changes; regenerate `tab:full_grid` after `results_nested.json` changes.

## Last valid result

Current output matches the Appendix C tables in `overleaf/arxiv/appendix.tex`. The validation leaderboard ranks the 203 configurations with 100% outer-test coverage; Logit Bias ALS (`lambda=0.1`, `r=2`) ranks first under both validation metrics.
