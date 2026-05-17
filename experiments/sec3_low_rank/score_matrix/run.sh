#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python experiments/sec3_low_rank/score_matrix/plot.py
