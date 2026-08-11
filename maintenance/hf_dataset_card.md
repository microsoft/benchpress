---
license: cdla-permissive-2.0
task_categories:
- tabular-regression
language:
- en
pretty_name: BenchPress Score Matrix
configs:
- config_name: scores_all
  data_files:
  - split: train
    path: data/scores_all.parquet
- config_name: scores_paper
  data_files:
  - split: train
    path: data/scores_paper.parquet
- config_name: models
  data_files:
  - split: train
    path: data/models.parquet
- config_name: benchmarks
  data_files:
  - split: train
    path: data/benchmarks.parquet
---

# BenchPress Score Matrix

This dataset contains the public model-by-benchmark score matrix used by
BenchPress. The release includes the lossless audited JSON, benchmark cost
evidence, flat model and benchmark metadata, one row per observed score, and
the paper-canonical dense subset used in the BenchPress experiments.

The source repository is
[`microsoft/benchpress`](https://github.com/microsoft/benchpress).

## Canonical artifacts

`data/llm_benchmark_data.json` is the authoritative rich score-matrix artifact.
It preserves nested alternative candidates, audit provenance, source URLs, and
normalized benchmark cost fields that cannot be represented losslessly in CSV.

`data/benchmark_cost_evidence.json` is the authoritative raw public
cost-evidence artifact. `metadata.json` records the SHA-256 digest and byte size
of both canonical JSON files. The CSV and Parquet files are deterministic flat
exports from the score-matrix JSON.

## Files

| File | Contents |
|---|---|
| `data/llm_benchmark_data.json` | Lossless audited score matrix: models, benchmarks, scores, candidates, and audit provenance. |
| `data/benchmark_cost_evidence.json` | Raw public token, dollar, and run-budget evidence used by benchmark cost metadata. |
| `data/scores_all.csv` / `.parquet` | Flat numeric score rows in the audit pool. |
| `data/scores_paper.csv` / `.parquet` | Long-form rows for the paper-canonical matrix. |
| `data/models.csv` / `.parquet` | Model metadata and canonical evaluation settings. |
| `data/benchmarks.csv` / `.parquet` | Benchmark metadata and canonical benchmark settings. |
| `data/score_matrix_paper_wide.csv` | Wide model x benchmark matrix for the paper-canonical subset. |
| `data/README.md`, `data/SCHEMA.md` | Dataset conventions and the canonical JSON schema. |
| `data/LICENSE-CDLA-2.0.md` | Dataset license text. |
| `metadata.json` | Export counts, matrix construction metadata, file inventory, and canonical JSON hashes. |

## Quick start

```python
from datasets import load_dataset

scores = load_dataset("microsoft/benchpress-score-matrix", "scores_paper")["train"].to_pandas()
models = load_dataset("microsoft/benchpress-score-matrix", "models")["train"].to_pandas()
benchmarks = load_dataset("microsoft/benchpress-score-matrix", "benchmarks")["train"].to_pandas()
```

For the lossless audit artifact:

```python
import json
from urllib.request import urlopen

url = (
    "https://huggingface.co/datasets/microsoft/benchpress-score-matrix/"
    "resolve/main/data/llm_benchmark_data.json"
)
with urlopen(url) as response:
    matrix = json.load(response)
```

## Schema

The flat score tables include:

- `model_id`, `benchmark_id`, `score`
- `reference_url`, `source_type`, `audit_status`, `matches_canonical`
- `reported_setting_json`, `notes`

The lossless JSON additionally preserves `candidates`, audit-rule identifiers,
audit notes, timestamps, and benchmark-level cost evidence.

`models` and `benchmarks` include an `in_paper_matrix` flag that identifies
rows retained by the paper-canonical threshold filter.

## Matrix construction

The paper-canonical matrix applies the BenchPress construction pipeline:
audit-status filtering, canonical representative selection, and the iterative
threshold filter. Current export counts:

- audit pool: @@N_MODELS@@ models, @@N_BENCHMARKS@@ benchmarks, @@N_SCORES_ALL@@ score rows
- paper matrix: @@PAPER_MODELS@@ models x @@PAPER_BENCHMARKS@@ benchmarks,
  @@PAPER_OBSERVATIONS@@ observed cells (@@PAPER_FILL_PERCENT@@% fill)

## License

The dataset files are released under
CDLA-Permissive-2.0. The source repository's MIT license applies to code and
documentation, not to this dataset license grant.

## Caveats

Scores come from heterogeneous public sources: model cards, official blogs,
technical reports, benchmark leaderboards, and third-party aggregators. Each
score retains source and audit metadata so downstream users can choose their
own filtering policy.
