# Appendix C.2 Full Method Comparison Table

## Paper mapping

- Appendix: `app:method_comparison`.
- Tables: `tab:full_grid` (`gen_full_table.py`), `tab:model_selection` (`gen_selection_table.py`).

## Purpose

This directory generates the validation-error and outer-test-error leaderboards for Appendix C.2. It is a read-only derivative of the main §4.2 method comparison run. It also hosts the unit-scale and tail-error audit for the ICLR response, which reuses raw prediction rows from the method-comparison, probe-selection, and prospective experiments.

## Inputs

- `../../sec4_building_benchpress/method_comparison/manifest.json`: the canonical 329-configuration sweep used by the main-text leaderboard; `gen_selection_table.py` requires the validation shards to match it exactly, then retains the 203 configurations with 100% outer-test coverage.
- `../../sec4_building_benchpress/method_comparison/inner_scores/*.npz`: per-configuration inner-validation metrics used by `gen_selection_table.py`.
- `../../sec4_building_benchpress/method_comparison/results_nested.json`: per-pair outer-test leaderboard with hyperparameters selected on nested validation cells, used by `gen_full_table.py`.

## Outputs

- stdout: LaTeX `longtable` for `tab:full_grid`, or the Top-15 validation leaderboard for `tab:model_selection`.
- `unit_tail_metrics_summary.json`: summary of all-cell and percentage-only MedAE/MedAPE, P90 absolute error, and the fraction of cells with absolute error above 10 points for the headline BenchPress results and the logit-space model mean on the same cells.

## Run

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_c_sec4_methods/method_comparison
python gen_full_table.py
```

or:

```bash
bash run.sh
```

Unit-scale and tail metrics:

```bash
BENCHPRESS_DATA=/path/to/may.json PYTHONPATH=$PWD \
  python experiments/appendix_c_sec4_methods/method_comparison/summarize_unit_tail_metrics.py
```

## Resume / rerun

No experiment runs here. Regenerate `tab:model_selection` after `inner_scores/*.npz` or `manifest.json` changes; regenerate `tab:full_grid` after `results_nested.json` changes. Regenerate `unit_tail_metrics_summary.json` after any raw prediction file used by the method-comparison, all-known probe, or prospective experiment changes.

## Last valid result

Current output matches the Appendix C tables in `overleaf/arxiv/appendix.tex`. The validation leaderboard ranks the 203 configurations with 100% outer-test coverage; Logit Bias ALS (`lambda=0.1`, `r=2`) ranks first under both validation metrics.

Unit-scale and tail metric audit for the ICLR May matrix:

- code commit: `f2ce6da1f004d6dfd1799427bebd4c2c7d18566a`
- CHTC job: 6307838 (`ut_tailf`), 4 allocated CPUs, `python:3.11` container
- output: `unit_tail_metrics_summary.json`
- sources: method-comparison prediction shards 124 and 95, fixed all-known probe outputs at k=5, and `prospective_update/results/may_to_aug/raw_predictions.json.gz`
- headline MedAE all / MedAE percentage-only / MedAPE percentage-only / P90 all / P90 percentage-only / fraction above 10 points / n all / n percentage-only:

| Setting | Method | Values |
|---|---|---|
| Method comparison | `\benchpress{}` | 4.63 / 4.37 / 7.80% / 19.73 / 17.07 / 24.9% / 26,040 / 24,720 |
| Method comparison | logit-space model mean | 7.22 / 6.97 / 11.92% / 27.69 / 24.57 / 39.5% / 26,040 / 24,720 |
| Probe, cost-unaware k=5 | `\benchpress{}` | 4.74 / 4.52 / 7.93% / 19.73 / 17.83 / 25.6% / 2,354 / 2,249 |
| Probe, cost-unaware k=5 | logit-space model mean | 7.52 / 7.36 / 13.40% / 27.64 / 25.96 / 40.6% / 2,354 / 2,249 |
| Probe, cost-aware k=5 | `\benchpress{}` | 5.32 / 4.98 / 9.14% / 22.06 / 18.89 / 29.1% / 2,385 / 2,253 |
| Probe, cost-aware k=5 | logit-space model mean | 7.77 / 7.51 / 13.04% / 30.54 / 26.13 / 42.4% / 2,385 / 2,253 |
| Prospective new models, cost-unaware k=5 | `\benchpress{}` | 6.88 / 6.80 / 10.74% / 22.70 / 21.05 / 33.8% / 541 / 519 |
| Prospective new models, cost-unaware k=5 | logit-space model mean | 6.95 / 6.71 / 11.13% / 25.17 / 24.30 / 38.6% / 541 / 519 |
