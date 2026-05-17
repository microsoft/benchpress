# Five-shot Predictor Prompt Ablation

## Paper mapping

Section 5.3, "Why Not Ask an LLM?".

## Purpose

This prompt ablation tests a more natural LLM use case than the matrix-completer
prompt. Instead of giving the LLM the full sparse score matrix and asking it to
complete missing cells, each query asks for one target model's score on one
target benchmark, with five nearest peer models as in-context examples.

The experiment has two prompt conditions:

- `five_shot_named`: real model and benchmark names are visible.
- `five_shot_blind`: model and benchmark names are anonymized within each
  prompt, while scores and the five-shot example structure are preserved.

## Inputs

- Folds: `benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json`
- Matrix: `benchpress/data/llm_benchmark_data.json`
- BenchPress baseline: canonical default-prediction cache at
  `benchpress/evaluation/default_predictions/benchpress_default/predictions.npz`
- Shared evaluation/API utilities: `../shared.py`

For each held-out target cell `(model, benchmark)`, peer examples are selected
from the fold's training matrix only:

1. The target cell is hidden.
2. Candidate peer models must have an observed score on the target benchmark.
3. Candidate peer models must share at least five visible benchmarks with the
   target model.
4. The five peers with the highest Pearson correlation over shared visible
   scores are used as in-context examples.

## Outputs

- `results.json`: source of truth and resume cache.
- `predictions/`: ignored debug artifacts containing prompts and raw responses.

Each raw prediction stores the target cell, predicted score, true score, number
of shots, selected peer indices, and peer correlations. Metrics can be
regenerated without rerunning API calls.

## How to run

Do not run API experiments locally.

```bash
cd experiments/sec5_findings/llm_completer/five_shot_predictor
python run.py --models gpt-5.5 --fold-ids 0 --cell-limit 4 \
  --conditions five_shot_named five_shot_blind --batch-size 4 --max-tokens 8192
```

For syntax-only development, use `python -m py_compile run.py plot.py`.

## Parallel execution

The runner is serial over folds and prompt conditions so `results.json` remains
auditable and resumable. The full run uses separate condition calls against the
same `results.json`: `five_shot_named` with `--batch-size 64 --max-tokens
16384`, and `five_shot_blind` with `--batch-size 16 --max-tokens 16384`. Each
API call batches multiple target cells, but each query remains a compact
five-shot score-prediction task rather than a full-matrix completion task. To
keep API calls reliable, each query shows at most twelve known target scores and
four shared-score anchors per peer example.

## Resume / rerun

`results.json` is reusable only when its `protocol` exactly matches
`shared.PROTOCOL`. Each fold record must have matching `fold_id`, `seed`, and
`fold` metadata plus finite metrics and nonzero coverage; otherwise rerun that
fold with `--force`.

The BenchPress baseline is read from the canonical cache and copied into
`results.json["bp"]` for plotting/alignment. Do not recompute it as part of this
experiment.

Transient API failures (`429`, timeout, `409`, and `5xx`) are retried without
a fixed attempt cap using exponential backoff capped at five minutes. Non-
transient failures still stop the run so invalid credentials, malformed
responses, or code errors are not hidden.

## Last valid result

Pending. Run a one-fold remote smoke test first, then launch the full 30-fold
run if parsing, coverage, and token usage look correct.
