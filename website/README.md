# BenchPress · Interactive LLM Benchmark Predictor

Pick any LLM x benchmark cell to see the BenchPress point prediction, trust probability, and calibrated 90% prediction interval.

🌐 Live: https://microsoft.github.io/benchpress/

The score-matrix selector on the existing project page supports:

- the curated BenchPress paper snapshot;
- filtered HELM and Every Eval Ever (EEE) snapshots;
- a user-provided `scores.csv` with optional `scores.meta.json`;
- local in-browser completion and downloadable prediction CSVs.

Uploaded files stay in the browser. External and uploaded matrices show point
predictions and support counts, not the BenchPress-matrix calibrated trust
probability or 90% interval.

The downloadable template is linked directly beside the upload control and is
stored at `matrix/benchpress-score-matrix-template.zip`. Its CSV contract matches the
package interface: the first column is `model`, remaining columns are
benchmarks, and blank cells are missing scores. The optional metadata JSON
declares non-percentage metric types and ranges.

Links:

- Project page: https://microsoft.github.io/benchpress/
- Paper code: https://github.com/microsoft/benchpress
- Site source: https://github.com/microsoft/benchpress/tree/main/website
- Dataset: https://huggingface.co/datasets/microsoft/benchpress-score-matrix
- Paper: https://arxiv.org/pdf/2606.24020

Source matrix: 129 models x 253 benchmarks with 4,905 reported scores
(August 26, 2026 snapshot). Point predictions use Logit Bias ALS (rank 2,
lambda=0.1). Trust probabilities and intervals use the Section 4.4 hybrid
uncertainty model with conformal calibration. Trust probability estimates how
likely the prediction is to be within 10% of the benchmark's declared score
range, or 10 raw score units when no finite range is declared.

Generated prediction artifacts are intentionally not stored in Git. Rebuild
the default snapshot from the canonical score matrix as follows. If the matrix
identity changed, rerun all 329 Section 4.2 shards first; otherwise reuse the
existing identity-matched shards and start with confidence calibration.

```bash
experiments/sec4_building_benchpress/method_comparison/run.sh \
  --workers 48 --merge
python experiments/sec6_trust/confidence_calibration/run.py \
  --calibrator-path \
  benchpress/evaluation/default_confidence/benchpress_default/calibrator.pkl
python -m website.scripts.build_default_snapshot \
  --snapshot-date 2026-08-26
```

The shard runner automatically invalidates prediction files whose matrix or
ordered benchmark-metric identity differs. Confidence calibration and snapshot
generation then fail if any required shard is missing or belongs to another
matrix or metric specification.
