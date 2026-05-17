#!/bin/bash
# Appendix B: full transform x method grid table (companion to §4 top-10 in main body).
# Reads canonical results.json from sec4 method_comparison and emits an 84-row longtable.
# Prereq: sec4_building_benchpress/method_comparison/run.sh has populated results.json.
set -euo pipefail
cd "$(dirname "$0")"

python gen_full_table.py
