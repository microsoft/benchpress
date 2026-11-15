# Question-Level Prediction and Selection

Paper mapping: ICLR 2027 Appendix D.4 (`app:question_level_budgets`),
Figure `fig:question_prediction_selection`.

This directory owns the paper figure. Training, question selection, raw
predictions, and complete scoring summaries are documented in
`../../sec5_findings/item_level_probes/README.md`.

## Render or update

```text
Render the current paper figure
  → Run the command below from the repository root.
  → Read results.json, the derived projection of the complete scoring summaries.
  → Write figures/bp_question_prediction_selection.pdf and .png.
  → Inspect both panels and the compiled manuscript page.

Refresh the projection after an intentional source-summary change
  → Pass --accuracy-summary <path> and --selection-summary <path> together.
  → Require complete binary-accuracy grids.
  → Copy prediction rows for all six random-question budgets and selection
    rows for budgets 200/400/800; retain source configuration hashes and SHA256.
  → Verify each plotted mean and sample SD against its complete source summary.
```

```bash
python -m experiments.appendix_d_sec5_findings.item_level_probes.plot
```

Only saved summary values are plotted. Rendering runs locally on one CPU,
requires no inference, and can be repeated to replace the derived PDF/PNG.
The figure uses the shared visual identity and the original curve styles;
the manuscript supplies its caption.

## Inputs and provenance

| Panel | Complete scoring source | Configuration | Summary SHA256 |
| --- | --- | --- | --- |
| Prediction | `experiments/sec5_findings/item_level_probes/results/accuracy/0b10e5b35d18180a/summary.json` | `0b10e5b35d18180a` | `65f6259c904e09efc71ca6806389d5007096f856d479f9da4c41cfb90a1402a8` |
| Selection | `results/question_selection/a6d858760138c8e7/scores/d05067f579e71d47/summary.json` in the experiment's documented scored archive | `d05067f579e71d47` | `3638ba9124e3ef0f1f0cfc90e9a8dc859210e81c2c6339c8d88eb5ba0972911e` |

The complete runs used 425 models and 21,176 questions, with 298 training
models and 127 held-out models. Points are mean binary prediction accuracy
over five selection seeds; bands are one sample standard deviation.
Predictor results were scored on CHTC at source `9efee67`; the three-policy
selection result was scored on CHTC at source `19d613d`. The original experiment
README records the raw archives, job IDs, and numerical environments.

Within each panel and seed/budget, all methods share the same hidden
model-question pairs. The selection panel excludes questions revealed by
any of its three policies. Its GP-IRT selection has 399 unique questions in
two seeds at nominal budget 400. Each panel's evaluation population can
change with budget; the two panels use different masks.
