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

Source matrix: 84 models x 133 benchmarks. Point predictions use Logit Bias ALS (rank 2, lambda=0.1). Trust probabilities and intervals use the Section 4.4 hybrid uncertainty model with conformal calibration. Trust probability estimates how likely the prediction is to be within 10 score points of the true benchmark result.
