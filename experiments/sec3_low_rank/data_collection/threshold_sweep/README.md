# Threshold Sweep Table (§3.1)

## Paper mapping
- **Section**: §3.1 Data Collection
- **Table**: `tab:bp_threshold_sweep`
- **Source comment**: `github/experiments/sec3_low_rank/data_collection/threshold_sweep/gen_table.py`

## Purpose
This table shows how the iterated minimum-observation filter trades matrix size for density. It uses the same canonical matrix loader as the rest of the paper.

## How to run
```bash
cd ~/Documents/submission/benchpress/github
python experiments/sec3_low_rank/data_collection/threshold_sweep/gen_table.py
```

## Inputs
- `benchpress.build_benchmark_matrix.load_score_matrix`

## Outputs
- Printed LaTeX tabular for `tab:bp_threshold_sweep`. Paste it inside the existing Overleaf table wrapper; the script does not emit `\begin{table}`, caption, or label.

## Resume / rerun
No long-running computation. Rerun the script after matrix-construction changes and paste the output into Overleaf.

## Last valid result
- Current adopted setting: minimum 15 observations per model and 8 per benchmark.
- Resulting matrix: 84 models × 133 benchmarks, 2,604 observed cells, 23.3% fill.
