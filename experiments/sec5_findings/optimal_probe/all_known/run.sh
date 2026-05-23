#!/bin/bash
# Greedy probe-set selection. Run from any directory.
set -euo pipefail
cd "$(dirname "$0")"

# Pin BLAS to 1 thread per worker so ProcessPoolExecutor scales linearly.
# Without this, 4 workers each spawn ~6 BLAS threads and thrash on 24 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

METRIC="${METRIC:-medape}"
if [[ -n "${CANDIDATE_ALLOWLIST:-}" ]]; then
  DEFAULT_OUT="greedy_${METRIC}_candidates_allowlist.json.gz"
else
  DEFAULT_OUT="greedy_${METRIC}_candidates_all.json.gz"
fi
WORKERS="${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

args=(
  --max-steps "${MAX_STEPS:-10}"
  --workers "$WORKERS"
  --metric "$METRIC"
  --out "${OUT:-$DEFAULT_OUT}"
)

if [[ -n "${CANDIDATE_LIMIT:-}" ]]; then
  args+=(--candidate-limit "$CANDIDATE_LIMIT")
fi

if [[ -n "${CANDIDATE_ALLOWLIST:-}" ]]; then
  args+=(--candidate-allowlist "$CANDIDATE_ALLOWLIST")
fi

python run.py "${args[@]}"
