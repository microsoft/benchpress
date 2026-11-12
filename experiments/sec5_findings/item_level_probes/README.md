# Question-level budgets

## Paper mapping

- ICLR 2027 Appendix `\Cref{app:question_level_budgets}` (D.4, "Question-Level
  Budgets"); rebuttal R-ye3d Q1.

## Purpose

For each held-out model, the methods observe its answers on the same B
evaluated questions and predict its accuracy on the remaining (unevaluated)
questions of each of the 37 subtasks of the Open LLM Leaderboard v2. The
training models' answers to all questions are observed.

Two ways of choosing the evaluated questions, shared by all methods and fixed
across held-out models within a (seed, B):

| Selection | Evaluated questions in subtask j |
| --- | --- |
| `random` | `RandomSampling().fit(train answers, k_j, seed=1000*seed+j).get_coreset()` |
| `anchors` | distinct `GPIRTPred().fit(train answers, k_j, seed=1000*seed+j).anchor_points` (at most k_j) |

k_j splits B equally across subtasks, each capped at floor(size/2); leftover
budget goes to uncapped subtasks. Every subtask keeps at least 66 unevaluated
questions.

| Method | Predictor |
| --- | --- |
| `benchpress_r{2,4,8}` | `predict_logit_bias_als_scores` (rank 2, 4 or 8, lambda 0.1 for every rank) with one fit per (selection, seed, B, rank) on the training models plus all 127 held-out models (answers 0/100; each held-out model reveals only its evaluated questions, unevaluated entries missing); prediction = mean predicted answer on the unevaluated questions |
| `random_sampling`, `random_sampling_learn`, `aipw`, `gp_irt` | estimators of Zhang, Dorner and Hardt (2025), package `benchpred`, given the shared evaluated questions; their full-subtask estimate f is converted to the unevaluated questions as (N f minus the observed sum) / (N minus n) |

For `anchors`, Random-Sampling-Learn is refit on the anchor questions and AIPW
uses them as its coreset. For `random`, GP-IRT uses the random questions as
equally weighted anchors with its fitted IRT model and lambda.

## Data and split

- Source: `socialfoundations/benchmark-prediction` at
  `5b7a23e32ee738c69b4c408cb9746429a7fd0043`, `data/scores/{bbh,gpqa,ifeval,openllm_math,mmlu_pro,musr}.jbl`.
- `build_data.py` keeps models present in all six files: 425 models,
  21,176 questions, 37 subtasks. Output `results/data/oll_items.npz` plus
  `.manifest.json` (source commit and sha256).
- Split: seed 42, 70% selection models (298), 30% held-out models (127),
  mirroring `split_models_for_probe_validation`.
- Budgets 200, 400, 800, 1600, 3200, 6400; random seeds 0 to 4.

## Metric

MedAE (accuracy points) between predicted and true accuracy on the unevaluated
questions, pooled over the 127 x 37 held-out cells
(`compute_prediction_error(..., aggregation='pool')`). Within a selection, all
methods are graded on identical cells. `summary.json` reports per-seed values
and the mean and sample std across the five seeds.

## How to run

Stages run in order; each compute unit is a JSON file carrying the
`config_hash` of `results/manifest.json` (`benchpress` units: that of
`results/benchpress/manifest.json`). Existing units are skipped, units of
another config fail.

```bash
python build_data.py --source-dir <benchmark-prediction checkout>
python run.py init --code-commit "$(git rev-parse HEAD)"
python run.py estimators --seeds 0 --budgets 200 --columns 33 --code-commit "$(git rev-parse HEAD)"
python run.py benchpress --selections random --seeds 0 --budgets 200 --ranks 2 --code-commit "$(git rev-parse HEAD)"
python run.py summarize --code-commit "$(git rev-parse HEAD)"
python plot.py
```

`estimators` writes both selections' evaluated questions; `benchpress` reads
them, so it runs after all 37 columns of its (seed, B) exist. `estimators`
needs `benchpred` installed from the pinned commit (checked at
run time). On CHTC each job runs `chtc_stage.sh <stage> [options]` from the
AP's `~/bp_iclr/run_job.sh`; `chtc_stage.sh data` builds the data and runs
`init`, later jobs receive `results/` inputs as `ilp_inputs.tar.gz`.

## Outputs

- `results/manifest.json`: config (data sha256, code commit, budgets, seeds,
  split, selections, allocation, target, column sizes) and `config_hash`.
- `results/estimators/s{seed}/B{B}/c{column}.json`: evaluated questions per
  selection, raw GP-IRT anchor points, and each estimator's per-model
  predictions on the unevaluated questions.
- `results/benchpress/manifest.json`: BenchPress config (base `config_hash`,
  code commit, ranks, lambda, fit) and its own `config_hash`.
- `results/benchpress/{selection}/r{rank}/s{seed}/B{B}.json`: predictions for
  all held-out models on the unevaluated questions of each subtask.
- `results/summary.json`
- `figures/bp_question_level_budgets.pdf` via `save_fig`.

## Last valid result

The pooled logit runs at commit `5f63bd7` are complete: base configuration
`1313fe0f25c8bd88`, pooled configuration `0f1e6888f939810f`. Original estimator
outputs are reused without changes.

## Method-only variants

`variants.py` reads the existing data, split, estimator coresets, pooled
predictions, and summary. It never writes those inputs or changes `run.py`.
All variants use rank 8 and lambda 0.1, with no rank or lambda selection.

| Variant | Fitting method |
| --- | --- |
| `raw` | Original Bias ALS with the identity feature transform |
| `group_logit` | Logit Bias ALS with an additional model-by-subtask offset |
| `group_raw` | Identity-transform Bias ALS with that offset |
| `group_binary` | Bernoulli negative log likelihood with low-rank factors and that offset |
| `raw_shrink` | Raw ALS with training-estimated shrinkage of subtask residuals, without itemwise clipping |
| `task_mean_ridge` | Fixed-penalty ridge from all 37 revealed subtask means to hidden subtask means |

ALS retains column z-scoring, 40 iterations, and 10 initializations (42 to 51).
The grouped model is `mu + a_i + c_j + d_i,group(j) + U_i.V_j`; every parameter
except `mu` receives the same L2 coefficient. Bernoulli fitting works on 0/1
answers directly, using L-BFGS with at most 400 iterations per initialization,
`ftol=1e-7`, and `gtol=1e-4`. Optimizer status is retained, including any
iteration-limited fits. No method receives held-out unrevealed answers.
The default BenchPress predictor is unchanged.

```bash
python variants.py fit --methods raw group_logit group_raw group_binary \
  --selections random --seeds 0 --budgets 200 --code-commit "$(git rev-parse HEAD)"
python variants.py summarize --variant-hash <hash> --code-commit "$(git rev-parse HEAD)"
```

Outputs are isolated under `results/variants/<config_hash>/`: a manifest,
per-method/selection/seed/budget JSON predictions and optimizer diagnostics,
and compressed per-question predictions with checksums. Existing valid units
are reused. A damaged artifact or mismatched provenance fails rather than
silently recomputing. Summarization verifies every original per-seed MedAE
against the frozen summary to absolute tolerance `1e-12` and requires all
127 x 37 cells, then scores the variants with the same shared metric.

Run synthetic method checks remotely with
`python -m unittest discover -s tests -p test_question_variants.py`.
First run the four methods on one existing setting, verify artifacts and
observations, then fan out the remaining settings across CHTC CPU jobs.
No new baseline fits or question selections are needed.
The same selection/seed/budget/method flags can restrict `summarize` for
canary diagnostics; these write `pilot_summary.json` with `complete_grid=false`,
never the full-grid summary. Missing requested units still fail.

For `raw_shrink`, fully observed training rows estimate each subtask's
between-model residual-mean variance and within-model residual variance.
These determine empirical-Bayes shrinkage of a target model's revealed
residual mean. The correction accounts for the relationship between the
revealed and hidden subsets of a finite subtask. No hidden target answers
enter either fit or prior estimation.
`task_mean_ridge` learns a shared regression on training rows, using the same
revealed/hidden question partition to construct its supervised examples.
Features are standardized using training rows only; ridge alpha is 0.1.
It predicts means, not question-level probabilities, and is a separate
comparator rather than the default BenchPress method.
Each newly requested method set has its own manifest and artifact hash;
earlier method sets and their original source/configuration are preserved.
