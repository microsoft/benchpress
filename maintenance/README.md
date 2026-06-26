# BenchPress maintenance

This folder contains lightweight maintenance wrappers for the public
`microsoft/benchpress` repository. The goal is to make recurring matrix refreshes
repeatable without hiding which expensive steps will run.

## Quick use

```bash
# 1. Inspect what changed or is stale.
python maintenance/check_updates.py

# 2. Preview the refresh pipeline. This is the default and runs no heavy work.
maintenance/run_set.sh

# 3. Execute selected steps after reviewing the preview.
DRY_RUN=0 RUN_MATRIX=1 RUN_GREEDY=1 RUN_PLOTS=1 RUN_WEBSITE=1 maintenance/run_set.sh
```

Generated reports are written to `maintenance/reports/` and are intentionally
git-ignored.

## What is checked

- Raw score matrix: `benchpress/data/llm_benchmark_data.json`
- Matrix export: `benchpress/data/llm_benchmark_matrix.xlsx`
- Default prediction artifacts under `benchpress/evaluation/default_predictions/`
- Suggested probe sets from the Section 5.1 greedy/random scripts
- Ranking-aware suggested probe sets from the Section 5.2 greedy script
- Website data and prediction-interval post-processing

## Notes

- The raw JSON is not tracked in git. If it is missing, run
  `python -m benchpress.download_data` or place the audited matrix at
  `benchpress/data/llm_benchmark_data.json`.
- Greedy probe selection is CPU-heavy. The wrapper is dry-run by default and only
  runs those steps when `DRY_RUN=0 RUN_GREEDY=1`.
- Website `data.json` generation is not yet a single canonical script. The
  wrapper can run the existing interval post-processor after `data.json` has been
  refreshed.
