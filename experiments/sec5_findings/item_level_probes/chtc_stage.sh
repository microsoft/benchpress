#!/bin/bash
# CHTC job body for one run.py stage. The AP's generic run_job.sh clones this
# repository at the job commit, installs it, exports REPO/DATA, and runs
# cmds/<tag>.sh, which calls this script.
# Usage: chtc_stage.sh data | <run.py subcommand> [run.py options]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCR="${_CONDOR_SCRATCH_DIR:-$PWD}"
STAGE="$1"; shift
BENCHPRED_COMMIT=5b7a23e32ee738c69b4c408cb9746429a7fd0043
COMMIT="$(git -C "$HERE" rev-parse HEAD)"
# benchpress loads its packaged score matrix on import; this experiment does not use it.
export BENCHPRESS_DATA="$DATA/may.json"

if [ "$STAGE" = question-baselines ]; then
  python -m pip install -q --no-cache-dir -r "$HERE/requirements-baselines.txt"
else
  python -m pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
  # pygam is in the repo's requirement.txt but not in its pyproject dependencies.
  # Older progressbar2/python-utils avoid conflicting with py-irt's typing_extensions pin.
  python -m pip install -q --no-cache-dir "numpy==1.26.4" "pygam==0.12.0" \
    "progressbar2==4.2.0" "python-utils==3.5.2" \
    "benchpred @ git+https://github.com/socialfoundations/benchmark-prediction@${BENCHPRED_COMMIT}"
fi
python -m pip freeze | sed 's/^/pip: /'

cd "$HERE"
if [ -f "$SCR/ilp_inputs.tar.gz" ]; then tar xzf "$SCR/ilp_inputs.tar.gz"; fi
if [ "$STAGE" = data ]; then
  git clone -q https://github.com/socialfoundations/benchmark-prediction "$SCR/zhang"
  git -C "$SCR/zhang" checkout -q "$BENCHPRED_COMMIT"
  python build_data.py --source-dir "$SCR/zhang"
  python run.py init --code-commit "$COMMIT"
elif [ "$STAGE" = question-baselines ]; then
  cd "$REPO"
  if [ "${1:-}" = check ]; then
    python -m unittest discover -s tests -p test_question_baselines.py
    python -m unittest discover -s tests -p test_question_accuracy.py
  else
    python -m experiments.sec5_findings.item_level_probes.baselines "$@" --code-commit "$COMMIT"
  fi
else
  python run.py "$STAGE" --code-commit "$COMMIT" "$@"
fi
