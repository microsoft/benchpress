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

python -m pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
# pygam is in the repo's requirement.txt but not in its pyproject dependencies.
# python-utils>=3.8 needs typing_extensions>3.10.0.2, which py-irt pins; older
# progressbar2/python-utils keep pip from backtracking indefinitely.
python -m pip install -q --no-cache-dir "numpy==1.26.4" "pygam==0.12.0" \
  "progressbar2==4.2.0" "python-utils==3.5.2" \
  "benchpred @ git+https://github.com/socialfoundations/benchmark-prediction@${BENCHPRED_COMMIT}"
python -m pip freeze | sed 's/^/pip: /'

cd "$HERE"
if [ -f "$SCR/ilp_inputs.tar.gz" ]; then tar xzf "$SCR/ilp_inputs.tar.gz"; fi
if [ "$STAGE" = data ]; then
  git clone -q https://github.com/socialfoundations/benchmark-prediction "$SCR/zhang"
  git -C "$SCR/zhang" checkout -q "$BENCHPRED_COMMIT"
  python build_data.py --source-dir "$SCR/zhang"
  python run.py init --code-commit "$COMMIT"
else
  python run.py "$STAGE" --code-commit "$COMMIT" "$@"
fi
