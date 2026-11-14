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

### BenchPress question selection versus random

`selection.py` compares only two question-selection policies, with the same
rank-40 BenchPress predictor. It reuses the frozen rank-40 random-selection
predictions (`577115b7edd126a2`); it never refits random
or runs GP-IRT/AIPW. The split, budgets, five seeds, and per-subtask allocation
remain fixed.

One training-only rank-40 fit produces a fitted response profile for each
question across the 298 training models. Observed-answer restoration is
disabled in both ALS and the inverse transform. The profiles include all
latent coordinates and offsets through their fitted responses; latent factors
from different ensemble initializations are not averaged. No held-out answers
enter this selection fit.

Within each subtask, MiniBatchKMeans clusters these profiles into its allocated
number of questions: k-means++ initialization, three initializations, batches
of 1024, at most 100 iterations, ten non-improving batches before stopping,
and seed `1000*selection_seed+subtask`. Each center contributes its nearest
unused actual question, with index-order tie breaking. Duplicate profiles
therefore cannot reduce the allocated budget. Each prediction fit sees the training rows and all
127 held-out rows' selected answers, as in the frozen random runs.

For each seed and budget, both policies are scored on the same
questions: those unrevealed by either policy. `compute_binary_accuracy` uses
the existing strict `>0.5` rule. The old random scalar accuracies are not reused:
their raw predictions are rescored on this paired mask. Evaluation populations
can change across seeds and budgets; gains are paired within each setting.
Summary tables report mean accuracy, sample standard deviation, and the mean
and sample standard deviation of paired percentage-point gains. This remains
an exploratory comparison on the previously used fixed model split.

From the repository root, run `python -m
experiments.sec5_findings.item_level_probes.selection` with stages `profiles`,
`select`, `fit`, and `summarize`, each with `--code-commit <source-commit>`;
use `--ranks 40` for fitting and scoring.
Stages after `profiles` require `--selection-hash <hash>`; `--seeds`,
`--budgets` restrict units for canaries/resume. Training profiles and prediction
fits retain their original source manifest when reused. Scoring has its own
configuration hash, including evaluated ranks and the scorer commit, so
changing the evaluation scope cannot overwrite or relabel source artifacts.
Use Python 3.11.18 with `requirements-selection.txt`, matching the frozen
random fits' numerical-library versions. Outputs under
`results/question_selection/<hash>/` include the training-profile NPZ,
manifest, selected global question IDs and timings, per-setting prediction
NPZs/checksums, and exact evaluation question IDs. Scored outputs are in
`scores/<score-hash>/summary.json` (`pilot_summary.json` for an incomplete
seed/budget grid). Existing predictions, figures,
MedAE summaries, rebuttal, and Overleaf text remain unchanged.

### Binary question-prediction accuracy

`baselines.py summarize` scores the individual binary answers on unrevealed
questions of the 127 held-out models. A score strictly greater than 0.5
predicts 1; ties predict 0. Stored percentage scores are divided by 100.
`compute_binary_accuracy` pools all hidden model/question pairs, so larger
subtasks contribute more questions. Revealed answers and training rows are
excluded. Each figure point is the mean of five seed-level percentages;
bands are one sample standard deviation.

The four curves are BenchPress ranks 8 and 40, the IRT question predictor used
by GP-IRT, and the ridge question predictor used by AIPW. Each predicts a
separate value for every hidden question. The baseline labels explicitly
identify the internal predictors: the complete GP-IRT and AIPW estimators
target aggregate scores and are evaluated by the subtask-score protocol below.
Broadcasting their aggregate means is not supported for this comparison.

`baselines.py fit` calls the pinned upstream `PIRTPred.fit`,
`estimate_ability_parameters`, and `item_curve`. IRT parameters are fit once
per training split, selection seed, and subtask; these fits do not depend on
the question budget. The initial anchors must reproduce the frozen GP-IRT
anchors. Ability estimation uses the frozen selected questions, including
the upstream anchor multiplicities. Ridge uses the upstream AIPW settings
(`alpha=10`, intercept enabled), with training-model answers as features and
the held-out model's revealed answers as labels. Fitting the 127 targets as
multiple outputs shares the same linear system. The mean hidden prediction
must reproduce the original AIPW hidden estimate within `1e-8` percentage
points. The intercept makes the mean observed ridge residual zero.

Per-question predictions, IRT parameters, frozen-estimator checksums, target
order, question indices, seeds, package versions, and a normalized configuration
hash are retained under `results/question_baselines/<hash>/`. The baseline
runtime is pinned in `requirements-baselines.txt`.

The frozen rank-8 files contain subtask means. Its recovered per-question
NPZs are retained under configuration `8e7e521f9645ce06`; reuse these files.
If they are unavailable, rebuild only its sixty fits with
`variants.py fit --methods logit --rank 8`, preserving all original settings.
Accuracy scoring requires every
regenerated subtask mean to reproduce the frozen rank-8 prediction within
absolute tolerance `1e-12`. Other methods reuse their existing predictions.

```bash
python variants.py accuracy --variant-hashes 8e7e521f9645ce06 577115b7edd126a2 \
  --code-commit "$(git rev-parse HEAD)"
cd ../../..
python -m experiments.sec5_findings.item_level_probes.baselines fit \
  --columns 0 --seeds 0 --code-commit "$(git rev-parse HEAD)"
python -m experiments.sec5_findings.item_level_probes.baselines summarize \
  --baseline-hash <baseline-hash> --benchpress-summary <benchpress-accuracy-summary> \
  --code-commit "$(git rev-parse HEAD)"
cd experiments/sec5_findings/item_level_probes
python plot.py --summary results/accuracy/<accuracy-hash>/summary.json \
  --name bp_question_accuracy
```

Outputs are separate `results/accuracy/<config_hash>/summary.json`
files, recording the threshold, comparison/tie rule, aggregation, source
configurations, masks, prediction checksums, integer correct/total counts,
and seed-level accuracies. The scorer validates raw NPZ checksums, target
order, question indices, revealed answers, and exact mask agreement with the
reused BenchPress records. Fit supports `--columns`, `--selections`, `--seeds`,
and `--budgets` for a canary; summarization requires the complete frozen grid.
`chtc_stage.sh question-baselines fit` uses the pinned baseline runtime.
Scoring runs on one CPU after all baseline predictions are available.
Original MedAE files remain unchanged; no GPU is needed.

### Subtask-accuracy estimates

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
`577115b7edd126a2`. CHTC canary `8530299` passed the package tests and
frozen-reference reproduction. Controller `8530550` admitted the remaining
59 fits as grid `8530551`; summary job `8530646` verified all 60 settings,
reusing the canary. The complete comparison is reported below.
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
For shards stored in CHTC staging, use `file:///staging/...` input URLs and
`requirements = (TARGET.HasCHTCStaging == true)`; plain `/staging/...` paths
in `transfer_input_files` are rejected by the scheduler.

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

## Fixed-rank BenchPress results (2026-11-13)

Rank 40 reduces mean MedAE by 24.7% to 34.6% relative to rank 8 across all
12 selection/budget cells, but does not remove the large-budget plateau.
Error rises again at 6400 questions under both selections. Raw ALS with
shrunk residuals at rank 8 has lower mean error from 800 questions onward.

Only rank changed from the original pooled BenchPress fit: logit transform,
column standardization, lambda 0.1, 40 ALS iterations, ten initializations,
data, split, revealed questions, hidden targets, and scoring are unchanged.
All 60 settings completed. This extends the same exploratory model split,
not an independent validation split.

MedAE is the median absolute error over 127 held-out models times 37
subtasks (4,699 hidden-accuracy predictions), in accuracy points; lower is
better. Each cell below averages five question-selection seeds. The JSON
retains all per-seed values and their sample standard deviations.

### Random questions

| Method | 200 | 400 | 800 | 1600 | 3200 | 6400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BenchPress (rank 8) | 4.582 | 4.397 | 4.320 | 4.296 | 4.298 | 4.565 |
| BenchPress (rank 40) | 3.431 | 3.131 | 2.923 | 2.823 | 2.834 | 3.061 |
| Raw ALS + shrunk residuals (rank 8) | 3.608 | 3.158 | 2.792 | 2.552 | 2.475 | 2.348 |
| Random-Sampling-Learn | 3.973 | 3.376 | 2.675 | 2.222 | 1.966 | 1.928 |

### GP-IRT anchor questions

| Method | 200 | 400 | 800 | 1600 | 3200 | 6400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BenchPress (rank 8) | 4.508 | 4.308 | 4.214 | 4.205 | 4.243 | 4.533 |
| BenchPress (rank 40) | 3.394 | 3.091 | 2.834 | 2.774 | 2.773 | 2.998 |
| Raw ALS + shrunk residuals (rank 8) | 3.430 | 3.043 | 2.701 | 2.451 | 2.354 | 2.275 |
| Random-Sampling-Learn | 3.848 | 3.085 | 2.518 | 2.056 | 1.843 | 1.824 |

The actual anchor counts and seed interpretation are the same as above.
Baseline and rank-8 variant results were reused, not refitted.

### Runtime and artifacts

CHTC jobs requested one CPU and 8 GB RAM each, using Python 3.11,
NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, and pandas 3.1.1.
Median recorded fit time was 3,354.820 seconds, with a range of
2,429.201 to 7,839.841 seconds. These exclude scheduling and environment
setup; hosts varied. No GPU or new model inference was used.

All 60 raw archives are retained under
`CHTC:$STAGING/bp_iclr/question_rank40/`, with names
`out_variant_logit_r40_<selection>_s<seed>_B<budget>.tar.gz`.
The compact `summary.tar.gz` contains
`results/variants/577115b7edd126a2/{manifest,pilot_summary,summary}.json`.
The full summary has `complete_grid=true`, 60 fit diagnostics, and 480
per-seed metric rows including the seven frozen reference methods.
Its SHA256 is
`5ddf9dc4bac882452679eda2cfdb2b578b5f591e2c59eeade295414105fb48bc`.
It reproduces the same frozen reference-summary hash documented above.

`results/variants/rank40_comparison.json` is a derived merge of this summary
and the prior comparison. Its source hashes and method alias record that
`benchpress_r40` comes from `variant_logit` in configuration
`577115b7edd126a2`. It retains all 900 unique per-seed rows and 480 fit
diagnostics across the method explorations without modifying source outputs.
The focused figure uses:

```bash
python plot.py --summary results/variants/rank40_comparison.json \
  --name bp_rank40_comparison --methods benchpress_r8 benchpress_r40 \
  variant_raw_shrink random_sampling_learn
```

Lines show five-seed means; shaded bands show one sample standard deviation,
not confidence intervals. The default predictor, evaluation runner,
rebuttal, and Overleaf text are unchanged.

## Question-prediction artifact registry

Rank-8 recovery used source
`d58184dbc15ed91d4041b83eb0865f4863dedce7`, configuration
`8e7e521f9645ce06`, canary job `8564504`, and 59-job cluster `8565335`.
Every job exited successfully. All sixty recovered fits reproduce the frozen
subtask predictions within `1e-12`. Recorded fit times ranged from 636.8 to
2574.6 seconds. Each worker used one CPU; the canary requested 8 GB RAM and
the remaining workers requested 4 GB. The Python and package versions match
the fixed-rank run above.

The source for the 120 BenchPress accuracy records is
`results/accuracy/9dfdbee8f2ea1b35/summary.json`, with SHA256
`4ac12db82bf150ac260a6718a1c76d07a99db99801a83945d2062bdb66cc5937`.
The native-baseline comparison reuses only its `benchpress_r8` and
`benchpress_r40` records. Scoring job `8567764` at source `138c663`
validated the rank-8 reproduction and raw predictions. The frozen
reference-summary hash is
`fb2691b637170ceadbb820c414eaed6a270a3b57a3330a68836f23525581f89a`.

Persistent outputs are under `CHTC:$STAGING/bp_iclr/question_accuracy/`:
`summary.tar.gz` holds the compact result; sixty
`out_variant_logit_r8_<selection>_s<seed>_B<budget>.tar.gz` files hold recovered
rank-8 NPZs and manifests. The original rank-40 NPZs remain under
`CHTC:$STAGING/bp_iclr/question_rank40/`.

Native baseline recovery is complete: canary `8571386` and all 25 jobs in
cluster `8577715` exited successfully at source
`9efee67dc6b1b6c6495b2cefea129e643c6a781f`, configuration
`3629ce98b95b0040`. The canary passed three native-baseline tests and five
binary-accuracy tests. Its completed units were reused in the full grid.
The 26 archives contain exactly 185 IRT models and 2,220 prediction bundles,
with identical manifests and no duplicate units. Each bundle contains both
native predictors. Fits use Python 3.11.18 and `requirements-baselines.txt`;
the grid requests one CPU and 4 GB RAM per job, with no GPU or LLM inference.

All native archives remain under
`CHTC:$STAGING/bp_iclr/question_baselines/`:
`out_native_baselines_canary.tar.gz`,
`out_native_baselines_s<seed>_part<0..3>.tar.gz`, and
`out_native_baselines_s<seed>_c33.tar.gz`. Entries are rooted at
`results/question_baselines/3629ce98b95b0040/`.
Scoring job `8592083`, at the same source commit, completed on one CPU with
4 GB RAM. AP submission files are
`~/bp_iclr/job_native_{canary,grid,summary}.sub`.
Its compact `summary.tar.gz` in that staging directory contains
`results/accuracy/0b10e5b35d18180a/summary.json`, with SHA256
`65f6259c904e09efc71ca6806389d5007096f856d479f9da4c41cfb90a1402a8`.
The summary has 240 unique per-seed records and 48 five-seed cells. All
120 BenchPress records match their frozen source exactly; all four methods
share each setting's hidden mask, denominator, and observed-question count.

## Native question-prediction accuracy (2026-11-13)

Accuracy measures whether a predictor correctly identifies an unrevealed
question's binary answer outcome for a held-out model. Predict 1 only when
the probability is strictly greater than 0.5; ties predict 0. Accuracy is
`100 * correct / hidden model-question pairs`, so higher is better.
It is not the held-out LLM's own benchmark accuracy.

The matrix contains 425 models and 21,176 questions across 37 subtasks of
BBH, GPQA, IFEval, MATH, MMLU-Pro, and MuSR. The fixed split (seed 42) has
298 training models and 127 held-out models. Question-selection seeds 0
through 4, budgets 200 through 6400 below, and both frozen selection policies
are unchanged. Accuracy pools hidden model-question pairs, giving larger
subtasks more weight. The hidden question population changes with budget.
This remains the exploratory split used for method development, not a fresh
model holdout.

The baseline curves evaluate the internal IRT predictor of GP-IRT and ridge
predictor of AIPW, not their aggregate score estimators. They use native
question-specific predictions. BenchPress rank 8 has the highest mean at
200 and 400 questions under both selections; the ridge predictor has the
highest mean from 800 questions onward. Rank 40 exceeds rank 8 at 6400
questions, but both native baselines have higher mean accuracy there.

Each cell is mean accuracy (%) over five question-selection seeds.
The JSON and derived `table.csv` also retain sample standard deviations;
`per_seed.csv` retains the individual records.

### Random questions

| Method | 200 | 400 | 800 | 1600 | 3200 | 6400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BenchPress (rank 8) | 81.201 | 81.408 | 81.717 | 81.948 | 82.071 | 83.565 |
| BenchPress (rank 40) | 79.574 | 80.616 | 81.443 | 81.908 | 81.760 | 85.437 |
| IRT predictor (GP-IRT) | 70.674 | 70.253 | 71.593 | 79.039 | 83.076 | 85.705 |
| Ridge predictor (AIPW) | 77.427 | 80.408 | 82.759 | 84.046 | 84.715 | 86.819 |

### GP-IRT anchor questions

| Method | 200 | 400 | 800 | 1600 | 3200 | 6400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BenchPress (rank 8) | 81.244 | 81.607 | 81.853 | 81.991 | 82.187 | 83.815 |
| BenchPress (rank 40) | 79.788 | 80.692 | 81.578 | 82.110 | 82.004 | 85.668 |
| IRT predictor (GP-IRT) | 72.397 | 73.523 | 73.028 | 78.927 | 83.046 | 86.073 |
| Ridge predictor (AIPW) | 78.697 | 81.264 | 83.143 | 84.376 | 84.906 | 87.142 |

Generate the four-curve PDF and PNG with:

```bash
python plot.py --summary results/accuracy/0b10e5b35d18180a/summary.json \
  --name bp_question_accuracy
```

The figure shows means and one sample standard deviation, not confidence
intervals. Local delivery copies are in
`~/Downloads/benchpress-question-variants/`: `bp_question_accuracy.pdf`,
`bp_question_accuracy.png`, `native-accuracy-summary.json`,
`native-accuracy-table.csv`, and `native-accuracy-per-seed.csv`.
The aggregate-score results, rebuttal, and Overleaf text remain unchanged.
