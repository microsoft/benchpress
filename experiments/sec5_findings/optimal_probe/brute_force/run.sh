#!/usr/bin/env bash
# Brute-force exhaustive probe-set search. Run from any directory.
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python run.py "$@"
