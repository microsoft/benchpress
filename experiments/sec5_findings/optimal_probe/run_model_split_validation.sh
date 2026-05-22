#!/bin/bash
# Model-split validation for greedy probe-set selection. Run from any directory.
set -euo pipefail
cd "$(dirname "$0")"

# Pin BLAS to 1 thread per worker so ProcessPoolExecutor scales linearly.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

METRIC="${METRIC:-medae}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.7}"
SEED="${SEED:-42}"
if [[ -n "${CANDIDATE_ALLOWLIST:-}" ]]; then
  DEFAULT_OUT="model_split_validation_${METRIC}_train70_usercheap.json.gz"
else
  DEFAULT_OUT="model_split_validation_${METRIC}_train70_all.json.gz"
fi
WORKERS="${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

args=(
  --max-steps "${MAX_STEPS:-10}"
  --workers "$WORKERS"
  --metric "$METRIC"
  --train-fraction "$TRAIN_FRACTION"
  --seed "$SEED"
  --out "${OUT:-$DEFAULT_OUT}"
)

if [[ -n "${CANDIDATE_LIMIT:-}" ]]; then
  args+=(--candidate-limit "$CANDIDATE_LIMIT")
fi

if [[ -n "${MODEL_LIMIT:-}" ]]; then
  args+=(--model-limit "$MODEL_LIMIT")
fi

if [[ -n "${CANDIDATE_ALLOWLIST:-}" ]]; then
  args+=(--candidate-allowlist "$CANDIDATE_ALLOWLIST")
fi

python run_model_split_validation.py "${args[@]}"
