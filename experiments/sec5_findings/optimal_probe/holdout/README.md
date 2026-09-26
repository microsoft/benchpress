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

On the May `84 x 133` matrix, predictor-fit counts are:

- Any-benchmark greedy validation: `76,315` fits (`1,285` candidate prefixes x
  `59` train models, plus `500` validation fits).
- Low-cost greedy validation: `12,595` fits (`205` candidate prefixes x `59`
  train models, plus `500` validation fits).
- Random baseline: `2,500` fits (`10` k values x `10` seeds x `25` validation
  models).
