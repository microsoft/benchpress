# Threshold Sweep Tables (§3.1 / Appendix B.1)

## Paper mapping
- **Section**: §3.1 Data Collection and Appendix B.1 Data Collection
- **Tables**:
  - `tab:bp_threshold_sweep`
  - `tab:threshold_error_sweep`
- **Source comments**:
  - `github/experiments/sec3_low_rank/data_collection/threshold_sweep/gen_table.py`
  - `github/experiments/sec3_low_rank/data_collection/threshold_sweep/gen_error_table.py`
  - `github/experiments/sec3_low_rank/data_collection/threshold_sweep/error_results.json`

## Purpose
The main-text table shows how the iterated minimum-observation filter trades matrix size for density. The appendix table reports prediction error under the full threshold grid, demonstrating that the adopted `(15, 8)` setting is a coverage-density choice rather than the minimum-error configuration.

## How to run
```bash
cd ~/Documents/submission/benchpress/github
python experiments/sec3_low_rank/data_collection/threshold_sweep/gen_table.py
python experiments/sec3_low_rank/data_collection/threshold_sweep/gen_error_table.py
```

## Inputs
- `benchpress.build_benchmark_matrix.load_score_matrix`
- `error_results.json`: paper-facing aggregate from the canonical 10-seed × 3-fold threshold-error sweep.

## Outputs
- Printed LaTeX tabular for `tab:bp_threshold_sweep`. Paste it inside the existing Overleaf table wrapper; the script does not emit `\begin{table}`, caption, or label.
- Printed LaTeX tabular for `tab:threshold_error_sweep`.

## Resume / rerun
The table generators are local formatting steps over existing data. Do not rerun the prediction sweep unless the canonical matrix, folds, or default BenchPress predictor changes. The aggregate was produced from resumable per-configuration raw prediction shards under commit `108c0997a9083cbca80fdbd458f2878659f26b06` on GCR.

## Last valid result
- Current adopted setting: minimum 15 observations per model and 8 per benchmark.
- Resulting matrix: 84 models × 133 benchmarks, 2,604 observed cells, 23.3% fill.
- Across the canonicalized no-density-filter matrix and all nonempty threshold settings, MedAE ranges from 3.83 to 4.92 and MedAPE from 6.23% to 8.44%.
- The adopted setting has MedAE 4.63 and MedAPE 7.77%; several stricter filters have lower error but substantially fewer retained cells.
