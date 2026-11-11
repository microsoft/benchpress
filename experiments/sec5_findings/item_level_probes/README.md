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
| `questions` | BenchPress (`predict_logit_bias_als_scores`) on the training models plus the held-out model (answers 0/100, unevaluated entries missing); prediction = mean predicted answer on the unevaluated questions. Rank and lambda are tuned per selection and B (below) |
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

## BenchPress tuning

Rank in {2, 4, 8, 16, 32} and lambda in {0.1, 1, 10} (the BenchPress default
is rank 2, lambda 0.1) are chosen using training models only. `tune` treats
24 training models (seed 43 permutation of the training models) as
validation targets: each is predicted from the other 297 training models with
the seed 0 evaluated questions of the same selection and B. `select` keeps,
per selection and B, the grid point with the lowest validation MedAE (ties to
the smaller rank, then the smaller lambda); `questions` uses it for the
held-out models.

## Metric

MedAE (accuracy points) between predicted and true accuracy on the unevaluated
questions, pooled over the 127 x 37 held-out cells
(`compute_prediction_error(..., aggregation='pool')`). Within a selection, all
methods are graded on identical cells. `summary.json` reports per-seed values
and the mean and sample std across the five seeds.

## How to run

Stages run in order; each compute unit is a JSON file carrying the
`config_hash` of `results/manifest.json`. Existing units are skipped, units of
another config fail.

```bash
python build_data.py --source-dir <benchmark-prediction checkout>
python run.py init --code-commit "$(git rev-parse HEAD)"
python run.py estimators --seeds 0 --budgets 200 --columns 33 --code-commit "$(git rev-parse HEAD)"
python run.py tune --selections random --budgets 200 --ranks 2 --lams 0.1 --targets <tune rows> --code-commit "$(git rev-parse HEAD)"
python run.py select --code-commit "$(git rev-parse HEAD)"
python run.py questions --selections random --seeds 0 --budgets 200 --targets <held-out rows> --code-commit "$(git rev-parse HEAD)"
python run.py summarize --code-commit "$(git rev-parse HEAD)"
python plot.py
```

`estimators` writes both selections' evaluated questions; `questions` reads
them, so it runs after all 37 columns of its (seed, B) exist; `tune` needs
seed 0 and `questions` needs `select`. `estimators`
needs `benchpred` installed from the pinned commit (checked at
run time). On CHTC each job runs `chtc_stage.sh <stage> [options]` from the
AP's `~/bp_iclr/run_job.sh`; `chtc_stage.sh data` builds the data and runs
`init`, later jobs receive `results/` inputs as `ilp_inputs.tar.gz`.

## Outputs

- `results/manifest.json`: config (data sha256, code commit, budgets, seeds,
  split, selections, allocation, target, column sizes, tuning grid and tune
  models) and `config_hash`.
- `results/estimators/s{seed}/B{B}/c{column}.json`: evaluated questions per
  selection, raw GP-IRT anchor points, and each estimator's per-model
  predictions on the unevaluated questions.
- `results/tune/{selection}/B{B}/r{rank}_l{lambda}/m{model}.json`: validation
  predictions per grid point.
- `results/tune_selection.json`: validation MedAE of every grid point and the
  chosen rank and lambda per selection and B.
- `results/questions/{selection}/s{seed}/B{B}/m{target}.json`
- `results/summary.json`
- `figures/bp_question_level_budgets.pdf` via `save_fig`.

## Last valid result

Not run yet.
