# §5.3 Why Not Ask an LLM?

Active experiments: **matrix completer** and **five-shot predictor** prompt
ablations, each with informed/named versus blind variants.

## Paper mapping

- Main text: Section 5.3, "Why Not Ask an LLM?"
- Figures: `bp_llm_completer.pdf`, `bp_llm_five_shot_predictor.pdf`

## Purpose

Compare GPT-5.5 prediction of hidden benchmark scores under two prompt styles,
using the exact Section 4.2 folds:

- **Matrix completer**: give the full sparse score matrix and ask the LLM to
  complete hidden cells.
- **Five-shot predictor**: give one target score query with five nearest peer
  models as examples; no full score matrix is shown.

## What this directory contains

```
llm_completer/
├── README.md
├── run.sh
├── shared.py
├── five_shot_predictor/
│   ├── run.py
│   ├── run.sh
│   ├── plot.py
│   └── README.md
└── informed_vs_blind/
    ├── run.py
    ├── run.sh
    ├── plot.py
    ├── results.json
    └── figures/
        ├── bp_llm_completer.pdf
        └── bp_llm_completer.png
```

Retired experiments (`convergence`, `context_ablation`, and the old combined
three-panel plot) are not part of the active tree. Do not restore them unless
the paper text is changed back to a multi-experiment LLM section.

## Inputs

Both prompt families use the exact §4 method-comparison protocol:
`benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json` (`10` seeds ×
`3` folds, `base_seed=42`, `min_scores=1`). For each persisted fold, they hide
the fold's test cells and evaluate:

- `bp`: matrix-only BenchPress baseline on the same held-out cells.
- `informed.gpt-5.5`: matrix-completer GPT-5.5 with real model names,
  benchmark names,
  benchmark metadata, and matrix structure.
- `blind.gpt-5.5`: matrix-completer GPT-5.5 with anonymous labels and no
  metadata.
- `five_shot_named.gpt-5.5`: five-shot predictor GPT-5.5 with real model and
  benchmark names.
- `five_shot_blind.gpt-5.5`: five-shot predictor GPT-5.5 with anonymous labels.

The figure reports median MedAPE across the `30` fixed folds, matching the
§4 `medape_median` score-error convention.

## Parallel execution

The wrappers are serial over folds and prompt conditions. API calls are
resumable through `informed_vs_blind/results.json` and
`five_shot_predictor/results.json`; use fold-limited dry/smoke checks before
the full run rather than launching duplicate full jobs.

## Outputs

Each subexperiment's `results.json` is the source of truth and resume cache. It
stores one record per fold. The matrix-completer cache uses:

```json
{
  "protocol": {
    "name": "method_comparison_s10_f3_bs42_ms1",
    "fold_source": "benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json"
  },
  "bp": [{"fold_id": 0, "seed": 42, "fold": 0, "medape": 7.8, "n_total": 1392, "raw_predictions": []}],
  "informed": {"gpt-5.5": [{"fold_id": 0, "seed": 42, "fold": 0, "medape": 12.0, "coverage": 100.0, "tokens": {}, "raw_predictions": []}]},
  "blind": {"gpt-5.5": [{"fold_id": 0, "seed": 42, "fold": 0, "medape": 20.0, "coverage": 100.0, "tokens": {}, "raw_predictions": []}]}
}
```

Every `raw_predictions` entry is `{fold_id, seed, fold, i, j, actual,
predicted}`. The five-shot predictor additionally records `n_shots`,
`peer_indices`, and `peer_correlations`. These are the post-bottleneck
artifacts: metrics and plots can be regenerated without rerunning API calls.

Ignored `predictions/` batch caches may be produced while running API calls, but
they are not required for the paper once `results.json` contains raw per-cell
predictions.

## How to run

From this directory:

```bash
bash run.sh
```

This runs both active prompt ablations and regenerates their figures.

The full method-comparison run is much larger than the retired `K=5` diagnostic:
  `30` folds × `2` conditions × about `9` model batches per fold, or roughly
`540` API calls for GPT-5.5 with the current matrix-completer batch size.
The five-shot predictor also uses two prompt conditions, but sends cell-query
batches instead of repeating the full matrix context. The named condition can
use larger batches than blind; the active full-run wrapper uses `64` for named
and `16` for blind. Use a fold-limited smoke check before the full run:

```bash
python informed_vs_blind/run.py --models gpt-5.5 --fold-ids 0 --dry-run \
  --results-path /tmp/bp_llm_completer_dry_run.json
python five_shot_predictor/run.py --models gpt-5.5 --fold-ids 0 --dry-run \
  --results-path /tmp/bp_llm_five_shot_dry_run.json
python five_shot_predictor/run.py --models gpt-5.5 --fold-ids 0 --cell-limit 4 \
  --conditions five_shot_named five_shot_blind \
  --batch-size 4 --max-tokens 8192 \
  --results-path /tmp/bp_llm_five_shot_smoke.json
```

To regenerate only the figure from existing results:

```bash
python informed_vs_blind/plot.py
python five_shot_predictor/plot.py
```

Before adding another API model, configure the model through the OpenAI-compatible
client in `benchpress/call_model.py`,
add its display style in `benchpress/plot_helpers/visual_identity.py`, run a
one-fold dry/smoke check, and only then run the full folds.

## Resume / rerun

Resume source of truth is each subexperiment's `results.json`. A cached run is
reusable only when the full protocol object matches `shared.PROTOCOL`; invalid,
zero-coverage, or metadata-mismatched fold records must be dropped and rerun.

## Last valid result

Matrix-completer active result: GPT-5.5 only, 30 fixed folds from
`folds_s10_f3_bs42_ms1.json`, 84 x 133 matrix, `base_seed=42`, `min_scores=1`.
Five-shot predictor result: pending remote smoke/full run.
