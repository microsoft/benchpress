# BenchPress maintenance

This folder contains lightweight maintenance wrappers for the public
`microsoft/benchpress` repository. The goal is to make recurring matrix refreshes
repeatable without hiding which expensive steps will run.

## Quick use

```bash
# 1. Inspect what changed or is stale.
python maintenance/check_updates.py

# 2. Check README counts against the current matrix.
python maintenance/check_docs.py

# 3. Build the Hugging Face public table export locally.
python maintenance/export_hf_dataset.py

# 4. Preview the refresh pipeline. This is the default and runs no heavy work.
maintenance/run_set.sh

# 5. Execute selected steps after reviewing the preview.
DRY_RUN=0 RUN_MATRIX=1 RUN_GREEDY=1 RUN_PLOTS=1 RUN_WEBSITE=1 RUN_HF_EXPORT=1 maintenance/run_set.sh
```

Generated reports and local exports are written to `maintenance/reports/` and
`maintenance/exports/`, respectively. Both directories are intentionally
git-ignored.

## What is checked

- Raw score matrix: `benchpress/data/llm_benchmark_data.json`
- Matrix export: `benchpress/data/llm_benchmark_matrix.xlsx`
- Default prediction artifacts under `benchpress/evaluation/default_predictions/`
- Suggested probe sets from the Section 5.1 greedy/random scripts
- Ranking-aware suggested probe sets from the Section 5.2 greedy script
- Website data and prediction-interval post-processing
- README counts for raw models, benchmarks, score rows, and the paper matrix
- Hugging Face public export freshness under `maintenance/exports/hf_dataset/`

## Hugging Face export

`python maintenance/export_hf_dataset.py` writes the public dataset layout:

```text
maintenance/exports/hf_dataset/
├── data/
│   ├── models.csv
│   ├── benchmarks.csv
│   ├── scores_all.csv
│   ├── scores_paper.csv
│   └── score_matrix_paper_wide.csv
└── metadata.json
```

If `pyarrow` or `fastparquet` is installed, matching `.parquet` files are
written automatically. Upload is opt-in:

```bash
python maintenance/export_hf_dataset.py --upload
```

## Notes

- The raw JSON is not tracked in git. If it is missing, run
  `python -m benchpress.download_data` or place the audited matrix at
  `benchpress/data/llm_benchmark_data.json`.
- Greedy probe selection is CPU-heavy. The wrapper is dry-run by default and only
  runs those steps when `DRY_RUN=0 RUN_GREEDY=1`.
- Website `data.json` generation is not yet a single canonical script. The
  wrapper can run the existing interval post-processor after `data.json` has been
  refreshed.
