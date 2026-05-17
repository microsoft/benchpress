# Complete-Submatrix SVD (§3.3)

## Paper mapping
- **Section**: §3.3 Low-Rank Structure of the Score Matrix
- **Table**: `tab:submatrix` (main_body.tex)

## Purpose
Check whether dense, fully observed submatrices already show low effective rank before any imputation. Find complete model-by-benchmark blocks, mean-center each benchmark column, compute the singular spectrum, and report stable rank plus cumulative variance explained by the first one and two components.

## How to run
```bash
cd experiments/sec3_low_rank/low_rank_structure/submatrix_svd
python generate_data.py    # produces both JSONs in one pass
python gen_table.py        # prints LaTeX for tab:submatrix
```

Deterministic and cheap — runs single-machine.

## Inputs
- `benchpress.evaluation_harness.M_FULL`, `MODEL_IDS`, `BENCH_IDS`
- Current filtered score matrix from `benchpress/data/llm_benchmark_data.json`
- Submatrix search parameters in `generate_data.py`:
  - largest-submatrix diagnostic uses `min_benchmarks=5`
  - table sweep uses fixed benchmark counts `[4, 7, 10, 13]`

## Outputs
| File | Content |
|------|---------|
| `stable_rank_results.json` | Diagnostic provenance for the largest complete submatrix: model/benchmark ids, singular values, stable rank, cumulative variance |
| `submatrix_sweep.json` | Rows for `tab:submatrix`: benchmark count, model count, stable rank, top-1/top-2 variance |

## Last valid result
- **Matrix**: 84×133 (2,604 observed, 23.3% fill).
- Largest complete submatrix: 32 models × 5 benchmarks; stable rank 1.08; top-2 variance 96.5%.
- Fixed-count rows: 4×42, 7×11, 10×7, 13×6.
