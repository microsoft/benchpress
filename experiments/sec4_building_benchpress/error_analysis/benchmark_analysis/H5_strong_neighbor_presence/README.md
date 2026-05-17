# H5 Strong-Neighbor Presence

## Paper mapping
- Section: `sec:reliability_analysis`, benchmark-side Table `tab:predictability_factors` and Figure `fig:predictability_factors_51`.
- Parent runner: `../run.sh`.

## Purpose
H5 tests whether removing strongly correlated benchmark neighbors hurts prediction on a target benchmark. For each target benchmark, it masks full columns whose pairwise absolute Pearson correlation with the target is at least 0.95, 0.90, or 0.85, then compares against the paired no-mask baseline.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- `N_SEEDS = 5`, `SEED = 42`, thresholds `[0.95, 0.90, 0.85]`.

## Outputs
- `ablation_results.json`:
  - `records`: one record per `(benchmark, seed, threshold)`.
  - each record stores base/treatment MedAPE and MedAE, deltas, and paired `raw_base` / `raw_treat` predictions.
  - `wilcoxon`: benchmark-level paired Wilcoxon summaries, one median delta per benchmark.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/benchmark_analysis
python H5_strong_neighbor_presence/ablation.py
```

The script is deterministic but not sharded; preserve the existing output unless intentionally rerunning this ablation.
