# All-known ranking-preservation probe sets

## Paper mapping

- Appendix: `\Cref{app:ranking_preservation}`, table `tab:ranking_greedy_probe_set`.
- Hero Figure ranking panel uses these curves.

## Purpose

Choose probe prefixes that maximize same-benchmark pairwise ranking accuracy at
a five-point score margin under the current all-known-cell probe setting.

## How to run

```bash
cd experiments/sec5_findings/ranking_preservation/greedy_probe_set/all_known
WORKERS=48 ./run.sh

CANDIDATE_ALLOWLIST=../../../optimal_probe/candidate_allowlists/user_cheap_20260505.json \
  OUT=greedy_pairwise_margin5_top10_targets_usercheap_candidates_usercheap.json.gz \
  WORKERS=48 ./run.sh
```

Smoke test:

```bash
MAX_STEPS=1 CANDIDATE_LIMIT=2 WORKERS=2 OUT=smoke_pairwise_margin5.json.gz ./run.sh
```

## Inputs

- `evaluate_probe_set` and `compute_ranking_accuracy` from `benchpress.evaluation_harness`.
- Shared low-cost allowlist from `../../../optimal_probe/candidate_allowlists/`.

## Outputs

Results are under `results/`.

- `results/greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz`
- `results/greedy_pairwise_margin5_top10_targets_usercheap_candidates_usercheap.json.gz`

## Resume / rerun

Reruns resume from the completed trajectory and candidate cache only when the
objective, margin, protocol, candidate allowlist, candidate limit, and candidate
count match.

## Last valid result

- Cost-unaware k=10 current-matrix accuracy: 88.9%.
- Low-cost k=10 current-matrix accuracy: 86.2%.
