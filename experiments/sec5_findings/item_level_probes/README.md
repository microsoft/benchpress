# Question-level budgets

## Paper mapping

- ICLR 2027 Appendix `\Cref{app:question_level_budgets}` (D.4, "Question-Level
  Budgets"); rebuttal R-ye3d Q1.

## Purpose

Compare, at an equal budget B of evaluated questions per held-out model, how
well each method recovers that model's 37 subtask accuracies on the Open LLM
Leaderboard v2:

| Method | What the held-out model is evaluated on | Predictor |
| --- | --- | --- |
| `subtasks` | whole subtasks, chosen greedily by five-fold CV MedAE on the selection models among subtasks that fit the remaining budget | `predict_benchpress_scores` on the 37-column subtask matrix |
| `questions` | B random questions split equally across subtasks | `predict_benchpress_scores` on the model-by-question matrix (answers 0/100); subtask score = mean of observed and predicted answers |
| `random_sampling`, `random_sampling_learn`, `aipw`, `gp_irt` | the same per-subtask allocation as `questions` (GP-IRT picks its own anchors) | estimators of Zhang, Dorner and Hardt (2025), package `benchpred` |

Random-Sampling, Random-Sampling-Learn, AIPW and `questions` share the same
random coreset for each (seed, B, subtask).

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

MedAE (accuracy points) over all 127 x 37 held-out cells, revealed subtasks
counted as exact (`medae_all`), and over unrevealed cells only
(`medae_unrevealed`). Both use `compute_prediction_error(..., aggregation='pool')`.
`summary.json` reports per-seed values and the mean and sample std across seeds;
`subtasks` is deterministic and has one run.

## How to run

Stages run in order; each compute unit is a JSON file carrying the
`config_hash` of `results/manifest.json`. Existing units are skipped, units of
another config fail.

```bash
python build_data.py --source-dir <benchmark-prediction checkout>
python run.py init --code-commit "$(git rev-parse HEAD)"
python run.py select --workers 16 --code-commit "$(git rev-parse HEAD)"
python run.py subtasks --workers 16 --code-commit "$(git rev-parse HEAD)"
python run.py questions --seeds 0 --budgets 200 --targets <held-out rows> --code-commit "$(git rev-parse HEAD)"
python run.py estimators --methods gp_irt --seeds 0 --budgets 200 --columns 33 --code-commit "$(git rev-parse HEAD)"
python run.py summarize --code-commit "$(git rev-parse HEAD)"
python plot.py
python plot.py --metric medae_unrevealed
```

`estimators` needs `benchpred` installed from the pinned commit (checked at
run time). On CHTC each job runs `chtc_stage.sh <stage> [options]` from the
AP's `~/bp_iclr/run_job.sh`; `chtc_stage.sh data` builds the data and runs
`init`, later jobs receive `results/` inputs as `ilp_inputs.tar.gz`.

## Outputs

- `results/manifest.json`: config (data sha256, code commit, budgets, seeds,
  split, column sizes) and `config_hash`.
- `results/select/B{B}.json`, `results/select/cv_cache.jsonl`
- `results/subtasks/B{B}.json`
- `results/questions/s{seed}/B{B}/m{target}.json`
- `results/estimators/{method}/s{seed}/B{B}/c{column}.json`
- `results/summary.json`
- `figures/bp_question_level_budgets{,_unrevealed}.pdf` via `save_fig`.

## Last valid result

Not run yet.
