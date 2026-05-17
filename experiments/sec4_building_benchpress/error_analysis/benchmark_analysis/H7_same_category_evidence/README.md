# H7 Same-Category Evidence

## Paper mapping
- Section: `sec:reliability_analysis`, benchmark-side Table `tab:predictability_factors`.
- Parent runner: `../run.sh`.

## Purpose
H7 tests whether same-category benchmark evidence helps predict a target benchmark. For each target benchmark, it masks all other benchmarks in the same category and compares the treatment against the paired no-mask baseline.

## Inputs
- Canonical score matrix and benchmark categories from `benchpress.all_methods`.
- `N_SEEDS = 10`, `SEED = 42`.

## Outputs
- `ablation_results.json`:
  - `records`: one record per `(benchmark, seed)` after valid paired predictions.
  - each record stores category, number of same-category columns removed, base/treatment MedAPE and MedAE, deltas, and paired `raw_base` / `raw_treat` predictions.
  - `wilcoxon`: paired benchmark-level summary using one median delta per benchmark.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/benchmark_analysis
python H7_same_category_evidence/ablation.py
```

The script is deterministic but not sharded; preserve the existing output unless intentionally rerunning this ablation.
