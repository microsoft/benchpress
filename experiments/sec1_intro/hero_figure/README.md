# Hero Figure (§1 Intro, Figure 1)

## Paper mapping
- **Section**: §1 Introduction
- **Figure**: `figures/bp_hero_panel_a_examples.pdf` and `figures/bp_hero_panel_b_overall.pdf`
- **Overleaf target**: `figures/bp_hero_panel_a_examples.pdf` and `figures/bp_hero_panel_b_overall.pdf`

## Purpose
The current arXiv Hero Figure is a reproducible two-panel asset pair. `plot.py` renders the panels from committed summary JSON/JSON.GZ inputs into `figures/`; the paper-facing PDFs live in Overleaf `figures/`.

## Files
| File | Role |
|------|------|
| `run.py` | Data-provenance entry point for target-cell keep-k raw predictions; recommended on a remote CPU machine, not local |
| `run_target_cells.py` | ICLR 2027 panel A compute: target-hidden keep-k curves for selected cells on one score snapshot; remote CPU only |
| `plot.py` | Current arXiv two-panel renderer; reads committed summaries and writes generated PDFs to `figures/` |
| `results/target_cells_may_2026.json`, `results/target_cells_aug_2026.json` | `run_target_cells.py` outputs (config + manifest, per-seed raw predictions, per-k summary) for ICLR panel A |
| `results.json` | Legacy raw prediction rows from the older full keep-k sweep; retained only as provenance/fallback data, not as a current rendering target |
| `results/hero_candidate_grid_summary.json` | Source summary for the current four example cells in panel A |
| `results/phi4_reasoning_plus_gpqa_keepk_summary.json` | Source summary for the Phi-4 Reasoning Plus / GPQA Diamond cell in panel A |
| `results/current_hero_mask_ablation_k10.json.gz` | Raw predictions for the current example-cell keep-k curves |
| `results/current_hero_mask_ablation_k10_summary.json` | Per-panel median and interquartile summaries used to verify the current example-cell panel |
| `run.sh` | Full data-provenance command; recommended on a remote CPU machine, not local |
| `figures/` | Generated arXiv panel copies |

Removed older intermediate files: `ranking_vs_k_raw.json`, `hero_phase_data.json`, `regen_phase_data.py`, and `run_k_sweep_pod.py`. The active arXiv figure is rendered by `plot.py`.

## How to run

### Replot current paper Figure 1
```bash
cd experiments/sec1_intro/hero_figure
python plot.py
```

This renders and verifies the canonical current arXiv assets:
- `figures/bp_hero_panel_a_examples.pdf`
- `figures/bp_hero_panel_b_overall.pdf`

## Inputs
- `benchpress.evaluation_harness`: `M_FULL`, `MODEL_IDS`, `BENCH_IDS`
- `benchpress.all_methods.predict_benchpress_scores`
- `results/hero_candidate_grid_summary.json`
- `results/phi4_reasoning_plus_gpqa_keepk_summary.json`
- `results/current_hero_mask_ablation_k10_summary.json`
- `sec5_findings/optimal_probe/all_known/results/random_medape_hero_all_known.json.gz`
- `sec5_findings/optimal_probe/all_known/results/greedy_medae_targets_tall_candidates_tall.json.gz`
- `sec5_findings/optimal_probe/all_known/results/greedy_medae_targets_tall_candidates_usercheap.json.gz`
- `sec5_findings/ranking_preservation/greedy_probe_set/all_known/results/greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz`
- `sec5_findings/ranking_preservation/greedy_probe_set/all_known/results/greedy_pairwise_margin5_top10_targets_usercheap_candidates_usercheap.json.gz`
- `sec3_low_rank/data_collection/observation_pattern/figures/bp_matrix_clean_white.pdf` for the compact venue-specific matrix panel

## Outputs
`run.py` writes resumable keep-k raw predictions:
- `results.json`
- `results/current_hero_mask_ablation_k10.json.gz`
- `results/current_hero_mask_ablation_k10_summary.json`

`plot.py` writes rendered PDFs:
- `figures/bp_hero_panel_a_examples.pdf`
- `figures/bp_hero_panel_b_overall.pdf`

### ICLR 2027 panel A (target-hidden)
```bash
python run_target_cells.py --score-json <may.json> --output results/target_cells_may_2026.json \
  --target phi-4-reasoning-plus:gpqa_diamond --target deepseek-v4-pro:terminal_bench
python run_target_cells.py --score-json <aug.json> --output results/target_cells_aug_2026.json \
  --target gpt-5.6-sol:browsecomp --target claude-opus-5:arc_agi_1
python plot.py --hidden-target-panel-a   # -> overleaf/iclr2027/figures/bp_hero_panel_a_examples.pdf
```
For each target cell and each (k, seed), the target model keeps k scores sampled from its other observed benchmarks; the target cell is never revealed. k=0 is the benchmark median over the other models. GPT-5.6 Sol and Claude Opus 5 (both released July 2026) use the August 26, 2026 snapshot; the other two cells use the May 2026 snapshot. Phi-4 Reasoning Plus has 11 observed paper benchmarks (below the 15-benchmark row threshold), so it is appended to the May matrix as an extra row. `run_target_cells.py` rewrites its output after each target and skips finished targets on restart; a config mismatch fails fast.

## Resume / rerun
- `run.py --k K` skips shard files in `--shard-dir` whose size is >100 bytes.
- To rerun one bad shard, delete `k{K}_s{S}.json` in the shard directory and re-run a single `(k, seed)` shard with `python run.py --k K --seed S --output <shard_dir>/k{K}_s{S}.json`.
- To rebuild `results.json` after shards finish, run `run.py --merge --shard-dir "$SHARD_DIR"`.
- To render and verify the arXiv Hero Figure assets into `figures/`, run `plot.py`.

## Last valid result
- **Matrix**: 84×133 (2604 observed, 23.3% fill).
- **Paper Figure 1**: `plot.py` writes current arXiv panel PDFs under `figures/`; copy paper-facing PDFs to Overleaf and validate the compiled Overleaf output.
- **Key interpretation**: panel A uses the selected-cell keep-k setting, where a random keep-k prefix can directly reveal the plotted target cell and produce zero-error drops. Panel B uses the current-matrix all-known probe-setting curves from `optimal_probe/all_known/results/`; held-out model-split curves live in `optimal_probe/holdout/results/` for robustness/update notes, not the active paper-facing hero panel.
- **ICLR 2027 panel B**: `BENCHPRESS_DATA=<may.json> python plot.py --hidden-only-panel-b` writes `overleaf/iclr2027/figures/bp_hero_panel_b_overall.pdf` from `fixed_order_medae_{any,low_cost}_hidden_only.json.gz` and the random `summary_non_probe_by_k_seed` (unrevealed cells only). k=5 / k=10 MedAE: cost-unaware 4.745 / 4.345, cost-aware 5.318 / 4.919, random (median over 10 seeds) 7.503 / 6.120.
