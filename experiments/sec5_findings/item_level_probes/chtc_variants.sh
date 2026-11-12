#!/bin/bash
# Called by the existing AP run_job.sh at an exact source commit.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCR="${_CONDOR_SCRATCH_DIR:-$PWD}"
STAGE="$1"; shift
COMMIT="$(git -C "$HERE" rev-parse HEAD)"
export BENCHPRESS_DATA="$DATA/may.json"
python -m pip freeze
if [ "$STAGE" = check ]; then
  cd "$REPO"
  python -m unittest discover -s tests -p test_question_variants.py
  python -m unittest discover -s tests -p test_prior_residual.py
else
  cd "$HERE"
  tar xzf "$SCR/ilp_inputs.tar.gz"
  python variants.py "$STAGE" --code-commit "$COMMIT" "$@"
fi
