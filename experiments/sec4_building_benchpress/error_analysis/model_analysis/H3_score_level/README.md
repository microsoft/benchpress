# H3 Score Level

## Paper mapping

- Section: `sec:reliability_analysis`, model-side Table `tab:model_hypotheses` and Figure `fig:error_hypotheses_52`.
- Parent runner: `../run.sh`.

## Purpose

H3 tests whether a model's median observed benchmark score predicts model-side BenchPress error. The feature is `med_true`.

## How to run

Do not run predictor experiments on the local Mac. Run this from the model-analysis directory on the remote CPU environment, or through the parent `../run.sh` when regenerating all model-side hypotheses:

```bash
cd experiments/sec4_building_benchpress/error_analysis/model_analysis
bash H3_score_level/run.sh
```

## Parallel execution

No internal parallelism. This leaf computes one deterministic hide-half-per-model analysis over `N_SEEDS = 10`; parallelism is at the parent-runner level across independent H directories if needed.

## Inputs

- Canonical score matrix and model metadata from `benchpress.all_methods`.
- Hide-half per-model holdout from `../_shared.py` with `N_SEEDS = 10` and `SEED = 42`.

## Outputs

- `results.json`: per-model rows, univariate Spearman tests, and `raw_predictions` with seeds, model indices, benchmark indices, actuals, and predictions.

## Resume / rerun

`analyze.py` now recomputes `results.json` every time it is run. Existing `results.json` is overwritten only by this leaf script; no shard cache is reused.

## Last valid result

Current protocol: 84 x 133 score matrix, model hide-half holdout, `N_SEEDS = 10`, `SEED = 42`, feature `med_true`.
