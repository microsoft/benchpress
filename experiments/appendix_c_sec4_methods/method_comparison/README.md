# Appendix C.2 Full Method Comparison Table

## Paper mapping

- Appendix: `app:method_comparison`.
- Tables: `tab:full_grid` (`gen_full_table.py`), `tab:model_selection` (`gen_selection_table.py`).

## Purpose

This directory generates the full transform-by-method leaderboard table for Appendix C.2. It is a read-only derivative of the main §4.2 method comparison run.

## Inputs

- `../../sec4_building_benchpress/method_comparison/results.json`: metric summary from the 7-transform by 12-method grid.
- `../../sec4_building_benchpress/method_comparison/manifest.json`: per-configuration metrics, used as the full-coverage ranking population.
- `../../sec4_building_benchpress/method_comparison/results_nested.json`: per-pair leaderboard with hyperparameters selected on nested validation cells.

## Outputs

- stdout: LaTeX `longtable` for `tab:full_grid`, or the `tabular` for `tab:model_selection`.

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

No experiment runs here. Rerun only after `sec4_building_benchpress/method_comparison/results.json` changes.

## Last valid result

Current output matches the Appendix C full method grid in `overleaf/arxiv/appendix.tex`, with Logit Bias ALS (`lambda=0.1`, `r=2`) as the selected full-coverage point predictor.
