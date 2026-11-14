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
  python -m unittest discover -s tests -p test_question_accuracy.py
else
  cd "$HERE"
  tar xzf "$SCR/ilp_inputs.tar.gz"
  if [ "$STAGE" = summarize ] || [ "$STAGE" = accuracy ]; then
    shopt -s nullglob
    for archive in "$SCR"/out_variant_*.tar.gz "$SCR"/out_shrink_summary.tar.gz \
        "$SCR"/out_task_mean_ridge_summary.tar.gz "$SCR"/out_task_item_pcr_grid.tar.gz; do
      [ -f "$archive" ] || continue
      tar xzf "$archive" -C "$REPO" --wildcards \
        'experiments/sec5_findings/item_level_probes/results/variants/*'
    done
  fi
  python variants.py "$STAGE" --code-commit "$COMMIT" "$@"
  if [ "$STAGE" = summarize ]; then
    find results/variants -type f \
      \( -name manifest.json -o -name summary.json -o -name pilot_summary.json \) \
      -print0 | tar --null -czf "$SCR/variant_summaries.tar.gz" -T -
  elif [ "$STAGE" = accuracy ]; then
    tar czf "$SCR/accuracy_summaries.tar.gz" results/accuracy
  fi
fi
