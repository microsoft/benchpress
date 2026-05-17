#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
python experiments/sec3_low_rank/data_collection/threshold_sweep/gen_table.py
