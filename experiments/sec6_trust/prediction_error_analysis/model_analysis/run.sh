#!/bin/bash
# Prediction Error Analysis: model-side H1-H9 + table/figures.
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python H1_model_size/analyze.py
python H2_model_type/analyze.py
python H3_score_level/analyze.py
python H4_rank2_expressibility/analyze.py
python H5_neighbor_quality/ablation.py
python H6_neighbor_evidence/ablation.py
python H7_family_peers/ablation.py
python H8_observation_count/ablation.py
python H9_temporal/analyze.py
mkdir -p tables
python gen_table.py > tables/model_hypotheses.tex
python plot_section52.py # -> figures/bp_error_hypotheses_52.{pdf,png}
