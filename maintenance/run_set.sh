#!/usr/bin/env bash
# Refresh matrix-derived BenchPress artifacts. Dry-run is the safe default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DRY_RUN="${DRY_RUN:-1}"
WORKERS="${WORKERS:-48}"
RANDOM_WORKERS="${RANDOM_WORKERS:-24}"

RUN_DOWNLOAD="${RUN_DOWNLOAD:-0}"
RUN_MATRIX="${RUN_MATRIX:-1}"
RUN_DEFAULT_PREDICTIONS="${RUN_DEFAULT_PREDICTIONS:-0}"
RUN_GREEDY="${RUN_GREEDY:-0}"
RUN_RANDOM="${RUN_RANDOM:-0}"
RUN_PLOTS="${RUN_PLOTS:-0}"
RUN_WEBSITE="${RUN_WEBSITE:-0}"

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

run_shell() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "0" ]]; then
    bash -lc "$*"
  fi
}

run_report() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@" || true
  fi
}

if [[ "$DRY_RUN" != "0" ]]; then
  echo "DRY_RUN=1: preview only. Set DRY_RUN=0 to execute."
fi

if [[ "$RUN_DOWNLOAD" == "1" ]]; then
  run python -m benchpress.download_data
fi

run python -m json.tool benchpress/data/llm_benchmark_data.json /tmp/benchpress_matrix_json_check.json
run_report python maintenance/check_updates.py

if [[ "$RUN_MATRIX" == "1" ]]; then
  run python -m benchpress.build_benchmark_matrix --excel benchpress/data/llm_benchmark_matrix.xlsx
fi

if [[ "$RUN_DEFAULT_PREDICTIONS" == "1" ]]; then
  run python -m benchpress.default_predictions
fi

if [[ "$RUN_GREEDY" == "1" ]]; then
  run_shell "cd experiments/sec5_findings/optimal_probe && OUT=greedy_medape_targets_tall_candidates_tall.json.gz MAX_STEPS=10 WORKERS=$WORKERS ./run.sh"
  run_shell "cd experiments/sec5_findings/optimal_probe && CANDIDATE_ALLOWLIST=candidate_allowlists/user_cheap_20260505.json OUT=greedy_medape_targets_tall_candidates_usercheap.json.gz MAX_STEPS=10 WORKERS=$WORKERS ./run.sh"
  run_shell "cd experiments/sec5_findings/ranking_preservation/greedy_probe_set && OUT=greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz MAX_STEPS=10 WORKERS=$WORKERS ./run.sh"
  run_shell "cd experiments/sec5_findings/ranking_preservation/greedy_probe_set && CANDIDATE_ALLOWLIST=../../optimal_probe/candidate_allowlists/user_cheap_20260505.json OUT=greedy_pairwise_margin5_top10_targets_usercheap_candidates_usercheap.json.gz MAX_STEPS=10 WORKERS=$WORKERS ./run.sh"
fi

if [[ "$RUN_RANDOM" == "1" ]]; then
  run_shell "cd experiments/sec5_findings/optimal_probe && python run_random.py --k-max 30 --n-seeds 10 --workers $RANDOM_WORKERS"
fi

if [[ "$RUN_PLOTS" == "1" ]]; then
  run_shell "cd experiments/sec5_findings/optimal_probe && python plot.py --compare --random-in random_medape_hero_all_known.json.gz --cheap-in greedy_medape_targets_tall_candidates_usercheap.json.gz --out bp_probe_evaluation_cost_unaware"
  run_shell "cd experiments/sec1_intro/hero_figure && python plot.py"
fi

if [[ "$RUN_WEBSITE" == "1" ]]; then
  run python website/scripts/add_prediction_intervals.py
  run python -m json.tool website/data.json /tmp/benchpress_website_data_check.json
fi

run_report python maintenance/check_updates.py
