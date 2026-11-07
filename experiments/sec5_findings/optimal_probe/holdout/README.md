# Held-out scorecard-recovery probe validation

## Paper mapping

- Robustness artifacts for `\Cref{sec:probe_selection}`.
- Current paper-facing Hero Figure panel B uses the full-matrix `all_known/`
  curves; this directory is for held-out sensitivity checks and update notes.

## Purpose

Validate probe-set selection on held-out model rows. The split is deterministic:
70% train models and 30% validation models with seed 42. Greedy probes are
selected on train rows; validation isolates each held-out target model so
BenchPress sees train rows plus that target model's probe scores, but not other
held-out rows.

## How to run

```bash
cd experiments/sec5_findings/optimal_probe/holdout

METRIC=medae OUT=model_split_validation_medae_train70_all.json.gz \
  MAX_STEPS=10 WORKERS=48 ./run_model_split_validation.sh

CANDIDATE_ALLOWLIST=../candidate_allowlists/user_cheap_20260505.json \
  METRIC=medae OUT=model_split_validation_medae_train70_usercheap.json.gz \
  MAX_STEPS=10 WORKERS=48 ./run_model_split_validation.sh

python run_model_split_random.py --k-max 10 --n-seeds 10 --workers 48

# Baselines on the same held-out cells and the same probes: logit-space model
# mean and offsets only (Logit Bias ALS rank 0).
python score_heldout_baselines.py \
  results/model_split_validation_medae_train70_all.json.gz \
  results/model_split_validation_medae_train70_usercheap.json.gz \
  --out results/heldout_baselines_s42.json.gz

# Low-cost pool under the documented <= $30 public-cost rule (drops MMLU-Pro):
CANDIDATE_ALLOWLIST=../candidate_allowlists/public_cost_le_30usd.json \
  METRIC=medae OUT=model_split_validation_medae_train70_cost_le_30usd.json.gz \
  MAX_STEPS=10 WORKERS=24 ./run_model_split_validation.sh
```

CHTC-clean checkout commands with the May matrix:

```bash
cd "$REPO/experiments/sec5_findings/optimal_probe/holdout"

BENCHPRESS_DATA="$DATA/may.json" \
  METRIC=medae OUT=model_split_validation_medae_train70_all.json.gz \
  MAX_STEPS=10 WORKERS="$NCPU" ./run_model_split_validation.sh

BENCHPRESS_DATA="$DATA/may.json" \
  CANDIDATE_ALLOWLIST=../candidate_allowlists/user_cheap_20260505.json \
  METRIC=medae OUT=model_split_validation_medae_train70_usercheap.json.gz \
  MAX_STEPS=10 WORKERS="$NCPU" ./run_model_split_validation.sh

BENCHPRESS_DATA="$DATA/may.json" \
  python run_model_split_random.py --k-max 10 --n-seeds 10 --workers "$NCPU"
```

Smoke test:

```bash
MAX_STEPS=1 CANDIDATE_LIMIT=2 MODEL_LIMIT=8 WORKERS=2 \
  OUT=smoke_model_split_validation.json.gz \
  ./run_model_split_validation.sh
```

## Inputs

- Score matrix and model split from `benchpress.evaluation_harness`.
- Shared candidate allowlists from `../candidate_allowlists/`.
- Predictor: `predict_benchpress_scores`.

## Outputs

Results are under `results/`.

- `results/model_split_validation_medae_train70_all.json.gz`
- `results/model_split_validation_medae_train70_usercheap.json.gz`
- `results/model_split_random_medae_train70.json.gz`
- `results/heldout_baselines_s42.json.gz`: per-cell BenchPress, model-mean and
  offsets-only predictions on the stored `validation_non_probe` cells, with
  per-k pooled MedAE/MedAPE and per-target win fractions.
- `results/model_split_validation_medae_train70_cost_le_30usd.json.gz`

Each validation result contains `trajectory[*].validation_non_probe` as the
primary held-out metric and `trajectory[*].validation_with_probe_zero` for the
compatibility denominator.

## Resume / rerun

The validation script resumes only when metric, candidate universe, split,
train fraction, seed, model limit, and candidate count match. Training candidate
caches are under `results/.candidate_cache/`.

## Last valid result

Split seed 42 (59 selection models, 25 held-out models), github commit `2f1ca06`, CHTC CPU jobs 6307250 (`b2_any`, any-benchmark, 16 CPUs, 10,552 s) and the earlier `b2_low` / `b2_rand` jobs (low-cost greedy, random baseline). Held-out non-probe MedAE:

| k | Any-benchmark greedy | Low-cost greedy | Random (median of 10 seeds) |
|---|---|---|---|
| 1 | 6.59 | 6.59 | 10.37 |
| 5 | 5.32 | 5.23 | 7.22 |
| 10 | 4.26 | 5.54 | 5.91 |

Both greedy searches are below the random median at every k from 1 to 10. Any-benchmark probes on the selection split: GPQA Diamond, Terminal-Bench 2.0, AIME 2024, LiveCodeBench, ARC-AGI-1, MMLU-Pro, Aider Polyglot (diff), HLE Text, BrowseComp, Codeforces Rating. Paper: ICLR App D.1 held-out paragraph and main-body transfer sentence in `sec:probe_selection`.


Same-cell baselines (`score_heldout_baselines.py`, protocol `heldout_baselines_v2`, CHTC CPU job 7925132, github commit `7996392`; `results/heldout_baselines_s42.json.gz`). MedAE at k = 1 / 5 / 10, with the fraction of held-out models on which BenchPress has the lower per-target MedAE:

| Ordering | BenchPress | Offsets only (rank 0) | Logit model mean |
|---|---|---|---|
| Any-benchmark greedy | 6.59 / 5.32 / 4.26 | 5.96 / 5.64 / 4.95 (24 / 48 / 64%) | 8.11 / 8.43 / 8.80 (76 / 84 / 96%) |
| Low-cost greedy | 6.59 / 5.23 / 5.54 | 5.96 / 5.78 / 5.44 (24 / 40 / 52%) | 8.11 / 9.10 / 8.88 (76 / 80 / 84%) |

Offsets only beats BenchPress at k = 1 and 2; on the any-benchmark ordering BenchPress is lower at every k from 3 to 10.

Low-cost pool without MMLU-Pro (`candidate_allowlists/public_cost_le_30usd.json`, 24 benchmarks; CHTC job 7925234, tag `rb_e4h`, commit `7996392`; `results/model_split_validation_medae_train70_cost_le_30usd.json.gz`, `results/heldout_baselines_cost_le_30usd_s42.json.gz`). Same split; selection-split probes: GPQA Diamond, MATH-500, Aider Polyglot (diff), HMMT Feb 2026, Aider Polyglot (whole), AIME 2026, tau2-bench Retail, Arena-Hard Auto, SciCode, AIME 2025. Held-out non-probe MedAE at k = 1 / 5 / 10: 6.59 / 6.30 / 6.01 (with MMLU-Pro 6.59 / 5.23 / 5.54); below the random median at k = 1 to 9, above at k = 10 (6.01 vs 5.91). Offsets only on the same cells: 5.96 / 6.13 / 5.65; logit model mean 8.11 / 8.40 / 9.07.

On the May `84 x 133` matrix, predictor-fit counts are:

- Any-benchmark greedy validation: `76,315` fits (`1,285` candidate prefixes x
  `59` train models, plus `500` validation fits).
- Low-cost greedy validation: `12,595` fits (`205` candidate prefixes x `59`
  train models, plus `500` validation fits).
- Random baseline: `2,500` fits (`10` k values x `10` seeds x `25` validation
  models).
