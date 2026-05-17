# Appendix B.2 Full Method Comparison Table

## Paper mapping

- Appendix: `app:method_comparison`.
- Table: `tab:full_grid`.

## Purpose

This directory generates the full transform-by-method leaderboard table for Appendix B.2. It is a read-only derivative of the main §4.2 method comparison run.

## Inputs

- `../../sec4_building_benchpress/method_comparison/results.json`: metric summary from the 7-transform by 12-method grid.

## Outputs

- stdout: LaTeX `longtable` for `tab:full_grid`.

## Run

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_b_sec4_methods/method_comparison
python gen_full_table.py
```

or:

```bash
bash run.sh
```

## Resume / rerun

No experiment runs here. Rerun only after `sec4_building_benchpress/method_comparison/results.json` changes.

## Last valid result

Current output matches the Appendix B full method grid in `overleaf/arxiv/appendix.tex`, with Logit Bias ALS (`lambda=0.1`, `r=2`) as the selected full-coverage point predictor.
