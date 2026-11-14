#!/bin/bash
# Container worker for the archived-snapshot experiment.
set -euo pipefail
SNAPSHOT_COMMIT="$1"
SNAPSHOT_MODE="$2"
SNAPSHOT_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
SNAPSHOT_OUTPUT="experiments/sec5_findings/prospective_update/results/random_reveals"
cd "$SNAPSHOT_ROOT"

save_outputs() {
    status=$?
    if [ -d "$SNAPSHOT_ROOT/repo/$SNAPSHOT_OUTPUT" ]; then
        tar -czf "$SNAPSHOT_ROOT/random_reveals_${SNAPSHOT_MODE}.tar.gz" \
            -C "$SNAPSHOT_ROOT/repo" "$SNAPSHOT_OUTPUT"
    fi
    exit "$status"
}
trap save_outputs EXIT

git clone -q https://github.com/microsoft/benchpress.git "$SNAPSHOT_ROOT/repo"
cd "$SNAPSHOT_ROOT/repo"
git checkout -q "$SNAPSHOT_COMMIT"
printf 'Code commit: %s\n' "$(git rev-parse HEAD)"
python -m pip install -q --no-cache-dir -e . pytest
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$SNAPSHOT_MODE" = run ] && [ -f "$SNAPSHOT_ROOT/random_reveals_check.tar.gz" ]; then
    tar -xzf "$SNAPSHOT_ROOT/random_reveals_check.tar.gz"
fi
if [ "$SNAPSHOT_MODE" = check ]; then
    BENCHPRESS_DATA="$SNAPSHOT_ROOT/may.json" python -m pytest -q \
        experiments/sec5_findings/prospective_update/test_random_reveals.py
fi
python -m experiments.sec5_findings.prospective_update.run_random_reveals \
    --train-json "$SNAPSHOT_ROOT/may.json" \
    --eval-json "$SNAPSHOT_ROOT/aug.json" \
    --out-dir "$SNAPSHOT_OUTPUT" --workers 16 --mode "$SNAPSHOT_MODE"
