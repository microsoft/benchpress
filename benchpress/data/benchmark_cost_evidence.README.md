# Benchmark Cost Evidence

This file stores public, mechanically extractable evidence about benchmark
evaluation cost. It is separate from `llm_benchmark_data.json`, which stores the
canonical score matrix and intrinsic benchmark metadata.

## Rules

1. Store raw source numbers, not only derived ratios.
2. Use only source tables with explicit numeric cost evidence: prompt tokens,
   completion tokens, dollar cost, run budget, or equivalent measurable units.
   Default generation caps such as `max_new_tokens * task_count` are not
   reported token usage; store them only as explicitly labeled budget evidence,
   never as observed prompt/completion tokens.
3. Keep qualitative claims out of numeric factors. Notes such as "hard" or
   "requires reasoning" can motivate further search but do not define cost.
4. Preserve the source table structure: header order, row path, anchor row, and
   extraction formula must be auditable from the cited URL. Use the most
   human-readable audit page as the primary `source_url`/`source_urls` entry;
   if exact numbers come from a separate raw artifact or embedded data payload,
   cite that as a supplemental data URL and spell out the relationship in
   `notes`.
5. `benchpress_id` means an exact row-level match. Use `candidate_benchpress_ids`
   when the source row is only a family/variant-level proxy.

## Current schema

```json
{
  "sources": [
    {
      "source_id": "helm_capabilities_v1_15_0",
      "source_name": "HELM Capabilities v1.15.0",
      "evidence_type": "multi_benchmark_token_table",
      "source_urls": ["..."],
      "table_path": "groups[1].rows",
      "header_order": [
        "Group",
        "Description",
        "Adaptation method",
        "# instances",
        "# references",
        "# prompt tokens",
        "# completion tokens",
        "# models"
      ],
      "columns": {
        "benchmark": 0,
        "instances": 3,
        "prompt_tokens": 5,
        "completion_tokens": 6,
        "models": 7
      },
      "anchor": {
        "source_benchmark_name": "IFEval",
        "benchpress_id": "ifeval",
        "total_tokens": 88583.418
      },
      "rows": [
        {
          "source_benchmark_name": "GPQA",
          "benchpress_id": "gpqa_main",
          "prompt_tokens": 53239.17,
          "completion_tokens": 148568.469,
          "total_tokens": 201807.639,
          "relative_to_anchor": 2.278
        }
      ]
    }
  ]
}
```

## Consumer policy

Cost-tier scripts may use this file as source-backed relative evidence. Field
names depend on the evidence type: HELM rows report `prompt_tokens` /
`completion_tokens`, while Artificial Analysis rows report `input_tokens` /
`reasoning_tokens` / `answer_tokens` / `output_tokens` and `total_cost_usd`.
Consumers should prefer exact `benchpress_id` matches and treat
`candidate_benchpress_ids` as audit leads, not automatic formula inputs.
Token/bill rows should be interpreted as model-inference evidence unless the
source explicitly reports judge, human-review, tool/environment, or
infrastructure costs. Do not treat token-only evidence as full benchmark cost.
Do not infer observed usage from harness limits or leaderboard context-window
settings: `max_new_tokens`, `max_model_len`, `length=...`, and similar fields
are budgets/configuration unless the source separately reports realized token
counts.
For model-row sources such as Artificial Analysis, preserve `source_model_name`
and `source_model_slug`; input tokens are the most comparable benchmark-length
signal, while output/reasoning tokens and dollar cost remain model/run specific.
