# Greedy-rank probe pruning

## Paper mapping

- Section: `\Cref{sec:probe_selection}` / Appendix probe-selection diagnostics.
- This is a candidate-set pruning diagnostic for the same all-known-cell
  protocol used by `../all_known/` and `../brute_force/`.

## Purpose

Use greedy search as an elimination signal, not as the final selector. The
runner derives candidate ranks from an existing all-known greedy result. For
each greedy context, lower candidate-set MedAE is a better rank. Candidates are
aggregated by average normalized rank across source greedy steps, then only the
requested top count or top fraction is kept.

The runner does not re-evaluate candidates. The source greedy result is the raw
prediction artifact; this directory only stores rank-pruning manifests and
allowlists.

## How to run

Derive the top-30 benchmark rank-pruned allowlist from the existing full all-known greedy
result:

```bash
METRIC=medae \
KEEP_COUNT=30 \
SOURCE_GREEDY_RESULT=../all_known/results/greedy_medae_targets_tall_candidates_tall.json.gz \
OUT=results/rank_pruning_medae_top30_count_from_existing_greedy.json \
ALLOWLIST_OUT=../candidate_allowlists/full_rank_top30_count_by_greedy_20260605.json \
./run.sh
```

By default `MAX_STEPS` is unset, so all source greedy steps are used.

## Inputs

- Source greedy result from `../all_known/results/`.
- Score-matrix metadata from `benchpress.evaluation_harness` for benchmark names
  and categories.

## Outputs

Results are written under `results/`.

```text
results/rank_pruning_<metric>_<keep-label>_from_existing_greedy.json
```

The result stores:

- source greedy steps and per-context candidate ranks;
- per-candidate average normalized rank;
- keep/remove decisions;
- optional generated allowlist metadata.

## Last valid result

- Source: `../all_known/results/greedy_medae_targets_tall_candidates_tall.json.gz`
- Source greedy trajectory length: 10 steps.
- Derived result: `results/rank_pruning_medae_top30_count_from_existing_greedy.json`
- Allowlist: `../candidate_allowlists/full_rank_top30_count_by_greedy_20260605.json`
- Setting: no fixed/protected probes, all 10 source greedy steps, keep top 30
  benchmarks by aggregate rank.
- Decision: keep 30 candidates and remove 103.
