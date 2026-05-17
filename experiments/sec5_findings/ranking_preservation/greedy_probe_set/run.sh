#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKERS="${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

args=(
  --max-steps "${MAX_STEPS:-10}"
  --workers "$WORKERS"
  --out "${OUT:-greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz}"
)

if [[ -n "${CANDIDATE_LIMIT:-}" ]]; then
  args+=(--candidate-limit "$CANDIDATE_LIMIT")
fi

if [[ -n "${CANDIDATE_ALLOWLIST:-}" ]]; then
  args+=(--candidate-allowlist "$CANDIDATE_ALLOWLIST")
fi

python run.py "${args[@]}"
