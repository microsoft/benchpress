# Build Benchmark Matrix

This package owns the score-matrix construction pipeline. The raw audited data
stays in `benchpress/data/llm_benchmark_data.json`; this code only decides which rows
and columns form the analysis matrix.

## Pipeline

```
original audit pool
  -> audit-status filter
  -> canonical representative selection
  -> iterative threshold filter
  -> final score matrix
```

## Commands

```bash
# Final paper matrix
python -m benchpress.build_benchmark_matrix

# Original audit pool before canonicalization / thresholding
python -m benchpress.build_benchmark_matrix --m-threshold 0 --b-threshold 0

# Original audit pool after canonicalization, before thresholding
python -m benchpress.build_benchmark_matrix --m-threshold 0 --b-threshold 0 --deduplicate

# Inspect canonical drop -> keep rules
python -m benchpress.build_benchmark_matrix --list-canonical-rules

# Export the selected matrix
python -m benchpress.build_benchmark_matrix --output scores.csv
```

## Matrix layers

| Layer | Command | Shape | Observed |
|---|---|---:|---:|
| Original audit pool | `load_score_matrix(m_threshold=0, b_threshold=0)` | 188 x 316 | 4,493 |
| Canonicalized pool | `load_score_matrix(m_threshold=0, b_threshold=0, deduplicate=True)` | 181 x 304 | 4,177 |
| Final score matrix | `load_score_matrix()` | 84 x 133 | 2,604 |

## Canonicalization rules

The rules select one canonical representative from near-duplicate benchmark
families, model setting variants, benchmark composite averages, or benchmark
subset variants. They do not edit the JSON, average scores, or rewrite
model/benchmark ids.

Decision rule:

```text
model variants with different mode/effort
  -> keep one row only; never fill/merge across modes

benchmark versions with the same metric scale
  -> fill missing canonical cells only; mark filled cells non-canonical

benchmark variants with a different metric scale
  -> drop as a separate column; never fill canonical percentage cells
```

LiveCodeBench v5 and v6 are fill rules: they fill a missing `livecodebench`
cell, mark the synthetic cell as non-canonical, and note that the source version
differs.

| Drop | Keep | Family | Reason |
|---|---|---|---|
| `deepseek-v3.2-speciale` | `deepseek-v3.2` | DeepSeek-V3.2 | Same base model; keep the default-effort row because it has broader coverage. |
| `kimi-k2-thinking` | `kimi-k2` | Kimi K2 | Same base model with a different mode; keep the broader-coverage default row. |
| `lfm2.5-1.2b-thinking` | `lfm2.5-1.2b-instruct` | LFM2.5-1.2B | Same base model with a different mode; keep the instruct/non-thinking row. |
| `codeforces_avg8` | `codeforces_rating` | Codeforces | Same task family; keep Codeforces Rating because it has broader coverage. |
| `codeforces_pass8` | `codeforces_rating` | Codeforces | Same task family; keep Codeforces Rating because it has broader coverage. |
| `livecodebench_pro` | `livecodebench` | LiveCodeBench | Elo/rating variant on a competitive-programming pool; do not use it to fill pass@1 percentage cells. |
| `livecodebench_v5` | `livecodebench` | LiveCodeBench | Fill missing standard LiveCodeBench cells only; mark as non-canonical because v5 is a different version. |
| `livecodebench_v6` | `livecodebench` | LiveCodeBench | Fill missing standard LiveCodeBench cells only; mark as non-canonical because v6 is a different version. |
| `tau1_bench_avg` | `tau_bench_airline/tau_bench_retail` | τ-bench | Composite average of existing domain benchmarks; keep per-domain benchmarks. |
| `tau2_bench_avg` | `tau2_bench_airline/tau2_bench_retail/tau2_bench_telecom` | τ²-bench | Composite average of existing domain benchmarks; keep per-domain benchmarks. |
| `apex_shortlist` | `matharena_apex_2025` | MathArena Apex | Subset/shortlist variant; keep MathArena Apex 2025 because it has broader coverage. |
| `healthbench_hard` | `healthbench` | HealthBench | Hard-subset variant; keep HealthBench because it has broader coverage. |
| `mmlu` | `mmlu_pro` | MMLU | Same benchmark family; keep MMLU-Pro because it has broader coverage. |
| `mmlu_redux` | `mmlu_pro` | MMLU | Same benchmark family; keep MMLU-Pro because it has broader coverage. |
| `global_mmlu_lite` | `mmlu_pro` | MMLU | Same benchmark family; keep MMLU-Pro because it has broader coverage. |
