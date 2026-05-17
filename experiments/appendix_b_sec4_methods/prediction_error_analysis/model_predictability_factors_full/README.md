# Model-Side Full Hypothesis Grid

## Paper mapping

- Appendix: `app:model_analysis`, paragraph `app:model_predictability_factors_full`.
- Figure: `fig:model_predictability_factors_full` (`bp_model_predictability_factors_full.pdf`).
- Source comment: `overleaf/arxiv/appendix.tex`.

## Purpose

This appendix figure expands the main-text model-side prediction-error figure into the full hypothesis grid. It shows all nine model-side hypotheses against both score-error metrics, using the same canonical Section 4.3 outputs as the main table and figure. The rendered layout splits H1--H5 and H6--H9 into left/right blocks so labels remain readable in the paper PDF.

## Inputs

- `sec4_building_benchpress/error_analysis/model_analysis/H1_*`--`H9_*/results.json`

## Outputs

- `figures/bp_model_predictability_factors_full.pdf`
- `figures/bp_model_predictability_factors_full.png`

## Run

```bash
cd ~/Documents/submission/benchpress/github/experiments/appendix_b_sec4_methods/prediction_error_analysis/model_predictability_factors_full
python plot_full.py
```

This script only redraws the appendix figure from existing Section 4.3 result files.
