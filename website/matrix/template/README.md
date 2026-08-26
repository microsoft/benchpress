# BenchPress score-matrix template

## Files

- `scores.csv` is required. Rows are models, columns are benchmarks, and blank
  cells are scores for BenchPress to predict.
- `scores.meta.json` declares each benchmark's metric type, valid range, and
  direction. Benchmarks omitted from this file default to percentage scores in
  `[0, 100]`.

## Rules

- The first CSV header cell must be exactly `model`.
- Model and benchmark identifiers must be non-empty and unique.
- Score cells must contain a finite number or be blank. Do not use `NA`, `-`,
  or formulas.
- Every model and benchmark must have at least one observed score.
- Rank-2 prediction requires at least 3 models, 3 benchmarks, and 3 observed
  cells.
- For stronger support, aim for at least 15 observed benchmarks per model and
  at least 8 observed models per benchmark.

Fill in the observed scores, leave unknown cells blank, and upload both files
at https://microsoft.github.io/benchpress/#picker.
