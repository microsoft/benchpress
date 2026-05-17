#!/bin/bash
# §1 Hero figure: target-cell keep-k provenance + arXiv panel render.
# Full compute updates raw/summary results; plot.py renders and verifies figures/.
# Full compute is slow (≈84 model × 20 k × 10 seed sweep); prefer running on a
# remote CPU machine and using plot.py locally.
set -euo pipefail
cd "$(dirname "$0")"

python run.py
python plot.py
