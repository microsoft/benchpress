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

Model-split MedAE validation:

- Any-benchmark k=5 held-out non-probe MedAE: 5.31; k=10: 4.38.
- Low-cost k=5 held-out non-probe MedAE: 5.60; k=10: 5.66.
