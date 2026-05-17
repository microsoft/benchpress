#!/bin/bash
# Appendix B: full 7H x 2-metric benchmark predictability_factors figure
# (companion to main-body Fig. 5, which shows the three jointly supported H's).
# Reads results.json from sec4_building_benchpress/error_analysis/benchmark_analysis/H*/.
set -euo pipefail
cd "$(dirname "$0")"

python plot_full.py
