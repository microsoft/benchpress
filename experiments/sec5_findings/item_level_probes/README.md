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

## Frozen reference result

The pooled logit runs at commit `5f63bd7` are complete: base configuration
`1313fe0f25c8bd88`, pooled configuration `0f1e6888f939810f`. Original estimator
outputs are reused without changes.

## Method-only variants

`variants.py` reads the existing data, split, estimator coresets, pooled
predictions, and summary. It never writes those inputs or changes `run.py`.
Low-rank fits use rank 8 and lambda 0.1. Ridge alpha is 0.1; item residuals
use at most eight principal components per subtask. No rank or penalty
selection is performed.

The same runner accepts a fixed `--rank` (default 8). The explicitly selected
`logit` method uses the original BenchPress predictor without residual
correction; it is not included in the default seven-method set. For the
requested rank-40 comparison:

```bash
python variants.py fit --methods logit --rank 40 --code-commit "$(git rev-parse HEAD)"
```

Only rank changes: lambda 0.1, 40 ALS iterations, ten initializations
(42 through 51), pooled fitting, the existing two selections, five seeds, and
six budgets remain fixed. Its 60 settings have a separate manifest and
artifact directory; original predictions and summaries are reused unchanged.
Run one setting first and reuse its artifact in the complete grid.

The rank-40 run uses source commit
`25d69878f9892db6bcd75149cdd31ab603fba995`, configuration
`577115b7edd126a2`. CHTC canary `8530299` (`random`, seed 0, budget 200)
passed the package tests and frozen-reference reproduction; fitting took
2,914.49 seconds. Its median absolute error over 4,699 hidden model/subtask
accuracies was 3.310 accuracy points, versus 4.498 for the frozen rank-8
result. This is a single-setting pilot, not a five-seed grid result.
Controller `8530550` (`~/bp_iclr/rank40.dag`) admitted the remaining 59 fits
as grid `8530551`. The dependent summary requires all 60 settings, reusing
the canary. Full-grid results are pending.
Raw archives go to `CHTC:$STAGING/bp_iclr/question_rank40/`, where
`$STAGING=/staging/y/yzeng58`; the final compact summary is `summary.tar.gz`
in that directory. This avoids adding the grid to the nearly full AP home.

| Variant | Fitting method |
| --- | --- |
| `raw` | Original Bias ALS with the identity feature transform |
| `group_logit` | Logit Bias ALS with an additional model-by-subtask offset |
| `group_raw` | Identity-transform Bias ALS with that offset |
| `group_binary` | Bernoulli negative log likelihood with low-rank factors and that offset |
| `raw_shrink` | Raw ALS with training-estimated shrinkage of subtask residuals, without itemwise clipping |
| `task_mean_ridge` | Fixed-penalty ridge from all 37 revealed subtask means to hidden subtask means |
| `task_item_pcr` | Subtask-mean ridge plus rank-8 principal-component regression on within-subtask revealed-item residuals |

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
`task_item_pcr` adds question-specific information without fitting unrestricted
target-model offsets: training-only ridge removes the group-mean signal from
the revealed item features, then a rank-8 PCA projection and ridge regression
predict the corresponding hidden-mean residual. The rank is capped by the
available item dimensions; every ridge alpha remains 0.1. Feature statistics,
PCA directions, and regression weights use training rows only.
Each newly requested method set has its own manifest and artifact hash;
earlier method sets and their original source/configuration are preserved.

For large CHTC summaries, transfer the frozen `ilp_inputs.tar.gz` and the
completed `out_variant_*.tar.gz` shards directly to the summary job.
`chtc_variants.sh summarize` merges their variant outputs in worker scratch,
then writes `variant_summaries.tar.gz` containing manifests and summaries.
Transfer this compact bundle and the job log back to the AP; retain the
original per-job archives as the raw prediction source. This avoids an
additional full merge, input tarball, and duplicate prediction archive in
the AP home directory.

## Exploratory results (2026-11-12)

Removing logit reduces error, but raw ALS still plateaus near three accuracy
points. Training-estimated residual shrinkage gives a decreasing curve.
Subtask-mean ridge with item residuals performs best among the seven variants:
it beats Random-Sampling-Learn at budgets 200 through 1600 under both
selections, while Random-Sampling-Learn has slightly lower mean error at
3200 and 6400. The ridge methods are separate comparators; the default
BenchPress predictor remains unchanged.

These are complete grids: seven methods, two selections, five seeds, and six
budgets, totaling 420 fitted settings. Methods were developed after inspecting
this fixed split, so this is exploratory evidence. All low-rank fits pool the
held-out models using revealed answers only. The original evaluation, splits,
coresets, hidden targets, and baseline predictions are unchanged.

MedAE is the median absolute error over the 4,699 held-out model/subtask
accuracies, in accuracy points; lower is better. A value of 2 means the median
cell is two points from its true hidden-question accuracy. Each table entry
is the mean of five per-seed MedAEs. The JSON summaries also retain their
sample standard deviations and every per-seed value.

### Random questions

| Method | 200 | 400 | 800 | 1600 | 3200 | 6400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BenchPress (rank 8) | 4.582 | 4.397 | 4.320 | 4.296 | 4.298 | 4.565 |
| Random-Sampling | 11.631 | 8.037 | 5.724 | 4.385 | 3.595 | 3.306 |
| Random-Sampling-Learn | 3.973 | 3.376 | 2.675 | 2.222 | 1.966 | 1.928 |
| AIPW | 9.024 | 5.768 | 3.995 | 2.845 | 2.246 | 2.048 |
| GP-IRT | 7.227 | 5.959 | 4.720 | 3.260 | 2.769 | 2.664 |
| Raw ALS | 3.557 | 3.239 | 3.079 | 3.009 | 3.139 | 3.143 |
| Logit ALS + subtask offsets | 8.215 | 6.378 | 5.063 | 4.412 | 3.929 | 4.163 |
| Raw ALS + subtask offsets | 7.402 | 5.475 | 4.065 | 3.266 | 2.806 | 2.620 |
| Bernoulli + subtask offsets | 5.903 | 5.324 | 4.440 | 3.422 | 2.775 | 2.491 |
| Raw ALS + shrunk subtask residuals | 3.608 | 3.158 | 2.792 | 2.552 | 2.475 | 2.348 |
| Subtask-mean ridge | 3.113 | 2.866 | 2.589 | 2.382 | 2.344 | 2.395 |
| Subtask-mean ridge + rank-8 item residuals | 2.972 | 2.653 | 2.332 | 2.091 | 1.995 | 1.955 |

### GP-IRT anchor questions

| Method | 200 | 400 | 800 | 1600 | 3200 | 6400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BenchPress (rank 8) | 4.508 | 4.308 | 4.214 | 4.205 | 4.243 | 4.533 |
| Random-Sampling | 9.257 | 6.683 | 4.714 | 3.647 | 3.391 | 3.656 |
| Random-Sampling-Learn | 3.848 | 3.085 | 2.518 | 2.056 | 1.843 | 1.824 |
| AIPW | 7.151 | 4.992 | 3.525 | 2.592 | 2.076 | 1.910 |
| GP-IRT | 6.038 | 5.063 | 4.019 | 2.706 | 2.148 | 2.005 |
| Raw ALS | 3.432 | 3.141 | 3.022 | 2.959 | 3.055 | 3.012 |
| Logit ALS + subtask offsets | 7.318 | 5.774 | 4.683 | 4.064 | 3.765 | 4.074 |
| Raw ALS + subtask offsets | 6.500 | 4.891 | 3.646 | 2.889 | 2.514 | 2.376 |
| Bernoulli + subtask offsets | 5.364 | 4.884 | 4.061 | 3.008 | 2.440 | 2.211 |
| Raw ALS + shrunk subtask residuals | 3.430 | 3.043 | 2.701 | 2.451 | 2.354 | 2.275 |
| Subtask-mean ridge | 3.053 | 2.824 | 2.527 | 2.345 | 2.213 | 2.266 |
| Subtask-mean ridge + rank-8 item residuals | 2.838 | 2.549 | 2.206 | 1.999 | 1.854 | 1.835 |

Anchor sets retain their original sizes: mean actual counts are 200, 399.6,
800, 1599.6, 3200, and 6399.8. Every method uses the same realized set within
each setting. The five seeds vary question selection, not the model split.

Unrestricted subtask offsets hurt small-budget accuracy. The Bernoulli
results are iteration-limited: all 600 initializations reached 400 iterations
without optimizer convergence. They describe the recorded finite-iteration
procedure, not a converged optimum.

### Runtime and artifacts

Fits ran on CHTC with one requested CPU, Python 3.11, NumPy 2.4.6,
SciPy 1.17.1, scikit-learn 1.9.1, and pandas 3.1.1. No GPU or new model
inference was used. Median recorded seconds per setting, excluding scheduler
and environment setup, were:

| Variant | Median seconds |
| --- | ---: |
| `raw` | 1033.216 |
| `group_logit` | 410.198 |
| `group_raw` | 436.743 |
| `group_binary` | 4773.987 |
| `raw_shrink` | 972.016 |
| `task_mean_ridge` | 0.488 |
| `task_item_pcr` | 0.703 |

CPU hosts varied, so these timings are operational measurements rather than
a controlled hardware comparison.

| Methods | Fit commit | Configuration | Summary commit | Summary job |
| --- | --- | --- | --- | --- |
| `raw`, `group_logit`, `group_raw`, `group_binary` | `d8fb664d6461ce7532a7379d5ec07a302bffd371` | `37f038ab59473936` | `de24e6d` | `8416977` |
| `raw_shrink` | `b9d1a979b521b217c75cec5b6045b64db93f16e3` | `3e45aded7c84be36` | `b9d1a97` | `8416813` |
| `task_mean_ridge` | `b9d1a979b521b217c75cec5b6045b64db93f16e3` | `84605c28993268f9` | `b9d1a97` | `8416765` |
| `task_item_pcr` | `b4d4f75d967fdc63a220a176282840a1cc4dcedd` | `6efd14d29c34e248` | `b4d4f75` | `8416775` |

On CHTC, paths below are relative to `~/bp_iclr/`:

- The first four methods' raw predictions are in
  `out_variant_<method>_<selection>_s<seed>_B<budget>.tar.gz`;
  `out_variants_summary.tar.gz` holds their compact summary and manifest.
- `out_shrink_summary.tar.gz`, `out_task_mean_ridge_summary.tar.gz`, and
  `out_task_item_pcr_grid.tar.gz` each contain their complete method grid,
  per-question arrays, summary, and figures.
- Per-unit paths inside the full archives start with
  `experiments/sec5_findings/item_level_probes/results/variants/<config_hash>/`.
  The compact first-four summary starts with `results/variants/`.
- The frozen common input is `inputs/ilp/variants/ilp_inputs.tar.gz`.

All four summaries have `complete_grid=true` and reproduce every original
per-seed score to absolute tolerance `1e-12`. Their common reference-summary
SHA256 is `fb2691b637170ceadbb820c414eaed6a270a3b57a3330a68836f23525581f89a`.
Question data SHA256 is
`6b67365fb38d65702f83ec0ddb8c0e059190ac9dc009a59aeb9aaeb55738ec2e`.

Local cached summaries live at `results/variants/<config_hash>/summary.json`.
`results/variants/comparison.json` is a derived merge with source hashes,
all 840 unique per-seed rows (seven original methods plus seven variants),
and 420 fit-diagnostic records. It retains the pre-existing rank-2 and rank-4
reference rows; no additional rank fits were run for this exploration.
`plot.py --summary results/variants/comparison.json` renders the comparison;
`--methods` selects displayed curves without changing the saved results.
The evaluation runner and shared metric are unchanged. Rebuttal and Overleaf
text are outside this method-only exploration.
