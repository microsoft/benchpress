#!/bin/bash
# Prediction Error Analysis: benchmark-side H1-H7 + table/figures.
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python H1_low_rank_fit/analyze.py
python H2_score_level/analyze.py
python H3_score_spread/analyze.py
python H4_target_coverage/ablation.py
python H5_strong_neighbor_presence/ablation.py
python H6_strong_neighbor_support/ablation.py
python H7_same_category_evidence/ablation.py
mkdir -p tables
python gen_table.py > tables/predictability_factors.tex
python plot_section51.py          # -> figures/bp_predictability_factors_51.{pdf,png}
python plot_section51_appendix.py # -> figures/bp_predictability_factors_51_<metric>.{pdf,png}
