#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python H2_model_type/analyze.py
