# Matrix Completer Prompt Ablation

## Paper mapping

Section 5.3, "Why Not Ask an LLM?".

## Purpose

Compare GPT-5.5 score completion when benchmark/model identities are visible
(`informed`) versus anonymized (`blind`) on the exact Section 4.2
method-comparison folds. This is the **matrix completer** prompt family: the LLM
sees a sparse model-by-benchmark score matrix and is asked to complete hidden
cells. The BenchPress baseline is evaluated on the same held-out cells.

## Inputs

- Folds: `benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json`
- Matrix: `benchpress/data/llm_benchmark_data.json`
- BenchPress baseline: canonical default-prediction cache at
  `benchpress/evaluation/default_predictions/benchpress_default/predictions.npz`
- Shared prompt/evaluation code: `../shared.py`

## Outputs

- `results.json`: resume cache and source of truth, with per-fold `raw_predictions`
- `figures/bp_llm_completer.pdf`
- `figures/bp_llm_completer.png`

## How to run

Use the parent directory command:

```bash
cd experiments/sec5_findings/llm_completer
bash run.sh
```

For debugging only, `run.py` supports fold-limited runs. Do not run API experiments locally.

## Parallel execution

No internal parallel worker pool. The script is serial over folds and prompt conditions so the API resume cache stays simple and auditable. Use fold-limited dry/smoke checks before a full remote/API run; do not start duplicate full jobs against the same `results.json`.

## Resume / rerun

`results.json` is reusable only when its full `protocol` object exactly equals `shared.PROTOCOL`. Each fold record must have matching `fold_id`, `seed`, and `fold` metadata plus finite metrics and nonzero coverage; otherwise rerun that fold. Per-batch prompt/response artifacts under `predictions/<condition>_foldXXX_<model>/` are saved incrementally, so interrupted folds resume by skipping batches whose target cells already have cached predictions.

For slow OpenAI-compatible API models, set `BENCHPRESS_OPENAI_REQUEST_TIMEOUT=<seconds>` before launching `run.py`; otherwise the shared client uses its default timeout.

The BenchPress baseline is read from the canonical cache and copied into `results.json["bp"]` for plotting/alignment. Do not recompute it as part of this experiment.

## Last valid result

The active paper result is GPT-5.5 only on 30 fixed folds: 10 seeds x 3 folds, `base_seed=42`, `min_scores=1`.
