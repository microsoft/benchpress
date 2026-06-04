# BenchPress · Interactive LLM Benchmark Predictor

Pick any LLM x benchmark cell to see the BenchPress point prediction, trust probability, and calibrated 90% prediction interval.

🌐 Live: https://microsoft.github.io/benchpress/

Links:

- Project page: https://microsoft.github.io/benchpress/
- Paper code: https://github.com/microsoft/benchpress
- Site source: https://github.com/microsoft/benchpress/tree/main/website
- Dataset: https://huggingface.co/datasets/microsoft/benchpress-score-matrix
- Paper: coming soon

Source matrix: 84 models x 133 benchmarks. Point predictions use Logit Bias ALS (rank 2, lambda=0.1). Trust probabilities and intervals use the Section 4.4 hybrid uncertainty model with conformal calibration. Trust probability estimates how likely the prediction is to be within 10 score points of the true benchmark result.
