#!/bin/bash
# Greedy-elimination pruning for all-known probe candidates.
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

METRIC="${METRIC:-medae}"
FIXED_PROBES="${FIXED_PROBES:-gpqa_diamond}"
PROTECTED_PROBES="${PROTECTED_PROBES:-gpqa_diamond,mmlu_pro}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

args=(
  --max-steps "${MAX_STEPS:-5}"
  --workers "$WORKERS"
  --metric "$METRIC"
  --fixed-probes "$FIXED_PROBES"
  --protected-probes "$PROTECTED_PROBES"
  --max-gain-abs "${MAX_GAIN_ABS:-0.05}"
  --max-gain-rel "${MAX_GAIN_REL:-0.01}"
  --threshold-mode "${THRESHOLD_MODE:-any}"
  --max-unique-model-coverage "${MAX_UNIQUE_MODEL_COVERAGE:-0}"
  --category-guard-top-n "${CATEGORY_GUARD_TOP_N:-1}"
)

if [[ -n "${OUT:-}" ]]; then
  args+=(--out "$OUT")
fi

if [[ -n "${CANDIDATE_LIMIT:-}" ]]; then
  args+=(--candidate-limit "$CANDIDATE_LIMIT")
fi

if [[ -n "${CANDIDATE_ALLOWLIST:-}" ]]; then
  args+=(--candidate-allowlist "$CANDIDATE_ALLOWLIST")
fi

if [[ -n "${SOURCE_GREEDY_RESULT:-}" ]]; then
  args+=(--source-greedy-result "$SOURCE_GREEDY_RESULT")
fi

if [[ -n "${ALLOWLIST_OUT:-}" ]]; then
  args+=(--allowlist-out "$ALLOWLIST_OUT")
fi

python run.py "${args[@]}"
