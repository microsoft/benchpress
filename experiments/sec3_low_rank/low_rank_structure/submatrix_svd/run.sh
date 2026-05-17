#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python generate_data.py
python gen_table.py
