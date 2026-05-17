# Observation Pattern Figure (§3.1)

## Paper mapping
- **Section**: §3.1 Data Collection
- **Figure**: `fig:bp_matrix`
- **Source comment**: `github/experiments/sec3_low_rank/data_collection/observation_pattern/plot.py`

## Purpose
This figure visualizes the observed/missing pattern of the adopted score matrix after canonicalization and threshold filtering.

## How to run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec3_low_rank/data_collection/observation_pattern
bash run.sh
```

## Inputs
- `benchpress.plot_helpers.data.OBSERVED`, which is loaded from the paper-canonical score matrix.

## Outputs
- `figures/bp_matrix_clean_white.pdf`
- `figures/bp_matrix_clean_white.png`

## Resume / rerun
No expensive computation. Rerun `bash run.sh` after matrix changes. The paper uses `bp_matrix_clean_white.pdf`; missing cells are white and observed cells use the BenchPress blue.

## Last valid result
- Matrix: 84 models × 133 benchmarks, 2,604 observed cells, 23.3% fill.
