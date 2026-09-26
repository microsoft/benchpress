# Raw/Logit Soft-Impute Rank Sweep (§3.3)

## Paper mapping
- **Section**: §3.3 Low-Rank Structure of the Score Matrix
- **Figure**: `figures/bp_rank_ucurve_raw_logit.pdf`
- **Claim supported**: rank 2 minimizes held-out error for raw- and logit-space Soft-Impute SVD.

## Purpose
This experiment evaluates Soft-Impute SVD at ranks 1 through 10 in two score spaces: raw scores and logit-transformed scores. It uses the shared evaluation folds, pools held-out cell predictions, and stores held-out predictions for every rank so MedAPE and MedAE can be recomputed without rerunning the SVD sweep.

## How to run
Recommended on a remote CPU machine (the SVD sweep is slow on a laptop):

```bash
cd experiments/sec3_low_rank/low_rank_structure/rank_sweep
python run.py
python plot.py
```

`run.sh` runs the same sequence and pins BLAS thread counts for CPU jobs.

## Parallel execution
The current script runs all ranks serially in one CPU job. If this becomes a bottleneck, split by method/rank and merge into the same `results.json` schema.

## Inputs
- `benchpress.evaluation_harness.M_FULL`, `OBSERVED`, `MODEL_IDS`, `BENCH_IDS`
- `benchpress.evaluation_harness.load_folds`
- `benchpress.methods.completers.complete_soft_impute`
- Fold setting in `run.py`: 10 seeds × 3 folds, base seed 42, `min_scores=1`
- Rank sweep: 1 through 10 for Soft-Impute; 0 through 10 for logit-space Bias ALS (`lambda=0.1`, the BenchPress family; rank 0 = offsets only), run with `python run.py --methods logit_bias_als` and stored under `results.json["logit_bias_als"]`

## Outputs
`results.json` contains:
```json
{
  "ranks": [1, 2, "..."],
  "methods": ["identity_svd", "logit_svd"],
  "matrix": {
    "n_models": 84,
    "n_benchmarks": 133,
    "n_observed": 2604,
    "model_ids": ["..."],
    "benchmark_ids": ["..."]
  },
  "folds": {"n_seeds": 10, "n_folds": 3, "base_seed": 42},
  "identity_svd": {
    "2": {
      "medape": 0.0,
      "medae": 0.0,
      "raw_predictions": [
        {"seed": 0, "fold": 0, "model_idx": 0, "benchmark_idx": 0, "true": 0.0, "pred": 0.0}
      ]
    }
  }
}
```

`plot.py` reads `results.json` and writes `figures/bp_rank_ucurve_raw_logit.pdf` and `.png`. Each curve uses pooled held-out MedAPE and marks the best rank with a red star.
`plot.py --compact` writes `figures/bp_rank_ucurve_raw_logit_compact.pdf` with a smaller canvas and larger relative fonts; the ICLR 2027 draft uses this copy in `overleaf/iclr2027/figures/`.

## Resume / rerun
The script resumes per method/rank. A cached entry is only reused if it contains `medape`, `medae`, and `raw_predictions`; older aggregate-only entries are intentionally rerun so the bottleneck held-out predictions are preserved. The script never regenerates shared folds; if fold loading fails, fix the shared fold artifact before rerunning this experiment.

## Last valid result
- **Matrix**: 84×133 (2,604 observed, 23.3% fill).
- **Key result**: rank 2 minimizes held-out error in both raw and logit score spaces.
- **Rerun**: latest rerun produced at the current commit on a remote CPU machine.

## Held-out-model robustness (app:rank_geometry)

`holdout_models.py` reruns the same Soft-Impute rank sweep under a stricter split
(reviewer rdwq): the newest 20% of models (by release date) are removed from the
training matrix as a block, and each is predicted from the older models plus two
thirds of its own scores, with 95% cluster-bootstrap intervals over the held-out
models. It reuses the shared harness (`load_folds`, `compute_prediction_error`,
`complete_soft_impute`). Result in `rank_sweep_holdout_models.json`; figure via
`plot_holdout_models.py` -> `bp_rank_ucurve_holdout_models.pdf` (Fig. app:rank_geometry).
Logit-space minimum stays at rank 2; raw-space ranks 1-3 overlap within CIs.

## BenchPress rank sweep (app:rank_geometry)

`python run.py --methods logit_bias_als` sweeps logit-space Bias ALS
(`lambda=0.1`) over ranks 0-10 on the same 30 folds. The ICLR run (CHTC CPU job
6307628, github commit `9ac9963`, May matrix) is stored locally as
`results_logit_bias_als.json` (gitignored; mirror `CHTC:bp_iclr/out_e3_rank.tar.gz`),
because its Soft-Impute entries are empty and it must not replace `results.json`.
`python bias_als_rank_bootstrap.py --results results_logit_bias_als.json` writes
`bias_als_rank_bootstrap.json`: pooled MedAE per rank and 95% model-cluster
bootstrap intervals (1,000 resamples, seed 0) for the MedAE difference against
rank 2. Pooled MedAE is 5.17 / 4.80 / 4.59 / 4.53 / 4.54 / 4.60 at ranks 0-5 and
5.11 at rank 10; rank 2 beats rank 0 by 0.58 [0.43, 0.72] and rank 1 by
0.21 [0.11, 0.31], while the intervals for ranks 3-6 against rank 2 contain zero.
