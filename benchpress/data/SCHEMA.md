# BenchPress Score Matrix Schema

Authoritative spec for the BenchPress score matrix. Every entry in `benchpress/data/llm_benchmark_data.json` MUST conform to this schema.

## Design principles

1. **One model = one canonical eval setting.** A single model_id corresponds to one fixed combination of (mode, effort, tools, sampling, judge, harness, prompt_style). All scores for that model in the matrix must be reported under this exact setting.
2. **Each score must be source-backed.** Every cell has a `reference_url` pointing to the primary source (official tech report, blog, or trusted leaderboard).
3. **Benchmarks fix only their intrinsic properties** (question set version, scale, modality). Everything about *how a model is run on the benchmark* (tools/judge/harness/prompt) is part of the model's canonical setting.
4. **Soft consistency check across benchmarks.** Cells in the same benchmark may technically have different judge/harness because that's part of model config — but the audit pipeline cross-checks and flags inconsistencies for human review.

## Schema

### models.json

```yaml
- id: claude-opus-4.6                      # unique
  name: "Claude Opus 4.6"
  provider: Anthropic
  release_date: "2026-04-22"
  params_total_M: null                     # null if undisclosed
  params_active_M: null
  architecture: null                       # "moe" | "dense" | null
  is_reasoning: true
  open_weights: false

  canonical_setting:
    mode: thinking                         # thinking | non-thinking | n/a
    effort: high                           # low | medium | high | max | n/a
    tools: none                            # none | code | web | all | benchmark-specified
    sampling: pass@1                       # pass@1 | avg-of-N | cons@N | maj@N
    judge: rule-based                      # rule-based | gpt-4o | claude-3.5-sonnet | ...
    harness: official                      # official | matharena | lm-eval | simple-evals | ...
    prompt_style: zeroshot-cot             # zeroshot | zeroshot-cot | few-shot | system-prompted
    temperature: 0.0                       # null if not specified
    context: default                       # default | 16k | 32k | 128k | 1m
    notes: ""
```

### benchmarks.json

```yaml
- id: aime_2025                            # unique, lowercase, snake_case
  name: "AIME 2025"
  category: Math
  source_url: https://maa.org/...
  num_problems: 30

  canonical_setting:
    version: "AIME-2025-I+II"              # exact problem set
    metric_type: pct                       # pct | elo | rating
    range: [0, 100]                        # for clipping
    higher_is_better: true
    multimodal_input: false
    notes: ""

  cost:                                  # optional benchmark-level cost evidence
    source_id: artificial_analysis_eval_token_cost_gemini_2_5_pro_anchor
    source_name: "Artificial Analysis per-evaluation token usage and cost with Gemini 2.5 Pro anchor"
    source_url: https://artificialanalysis.ai/evaluations/gpqa-diamond  # primary human-auditable citation
    source_data_url: null                     # optional raw artifact / embedded-data page for exact extraction
    source_benchmark_name: "GPQA Diamond"
    source_model_name: "Gemini 2.5 Pro"     # required when usage is model-row-specific
    source_model_slug: gemini-2-5-pro
    evidence_scope: observed model-inference totals for the source evaluation run
    tokens:
      prompt_tokens: null                # HELM-style input tokens, if reported
      completion_tokens: null            # HELM-style output tokens, if reported
      input_tokens: 246768               # AA-style input tokens, if reported
      output_tokens: 1373612
      reasoning_tokens: 977350
      answer_tokens: 396262
      total_tokens: 1620380
    total_cost_usd: 15.77526875          # null if source does not report dollars
    reported_items: null                 # only when source explicitly reports item count for this run
    reported_samples: null               # only when source explicitly reports samples/pass@k for this run
    relative_tokens_to_source_anchor: 1.0
    relative_cost_to_source_anchor: 1.0
    notes: "Inference-side token/cost evidence only; judge/tool/environment cost is separate."
```

### scores.json

```yaml
- model_id: claude-opus-4.6
  benchmark_id: aime_2025
  score: 93.5                              # PRIMARY score (used in paper / md)
  reference_url: https://anthropic.com/...

  reported_setting:                        # what the source actually used
    mode: thinking
    effort: high
    tools: none
    sampling: pass@1
    judge: rule-based
    harness: matharena
    prompt_style: zeroshot-cot
    temperature: 0.0

  matches_canonical: true                  # auto-computed: reported == model.canonical_setting
  source_type: official_blog               # official_blog | tech_report | leaderboard | third_party | model_card
  audit_status: verified                   # pending | verified | flagged | dropped
  notes: ""

  candidates:                              # optional: alt scores for the same (model, benchmark)
    - score: 92.8                          #   each candidate has its own setting/source
      reference_url: https://matharena.ai/...
      source_type: leaderboard
      reported_setting: {...}
      notes: "matharena, slightly different harness"
    - score: 94.0
      reference_url: https://artificialanalysis.ai/...
      source_type: third_party
      reported_setting: {...}
      notes: "AA aggregator"
```

**candidates[] semantics**:
- The top-level `score` / `reference_url` / `reported_setting` is the **primary** (used everywhere downstream).
- `candidates[]` stashes other source values seen for the same cell (e.g., a third-party value seen while auditing a different model's blog).
- When a higher-priority source (per source priority list below) is later verified, **promote** it to primary; the displaced primary moves into `candidates[]`.
- `candidates[]` may be empty / omitted. The primary itself is implicitly also in the candidate set.
- A cell with primary = `pending` but multiple `candidates` is still useful: at least we have something while we hunt for the canonical source.

**When to add a candidate (vs. ignoring)**:
- ✅ Auditing model A's blog and that blog reports a value for model B → if B's BP cell already has a different primary, **add as candidate to B**.
- ✅ Discovering a new source while looking for something else.
- ❌ Don't add the primary itself as a duplicate candidate. Don't add candidates that are clearly stale (model spec changed, benchmark version changed).


## Inclusion rules

A `(model, benchmark)` cell is included in the matrix iff:

1. ✅ `reference_url` exists and is reachable
2. ✅ `matches_canonical = true` (or `notes` explain why discrepancy is acceptable)
3. ✅ `audit_status = verified`

If multiple sources exist for the same `(model, benchmark)`, prefer in order:

1. official_blog (provider's own announcement)
2. tech_report (provider's tech report / paper)
3. model_card (provider's HF model card)
4. leaderboard (trusted leaderboards: matharena, livebench, lmsys, lisanbench)
5. third_party (independent eval, e.g., artificialanalysis.ai)

## Sampling vocabulary

- `pass@1`: single roll-out per question, no averaging. Default.
- `avg-of-N`: average accuracy over N rollouts (e.g., `avg-of-32` for AIME).
- `cons@N`: consensus / self-consistency over N rollouts (majority vote).
- `maj@N`: majority vote over N rollouts (synonym for cons@N).
- `best-of-N`: pick best of N (rare in benchmark reports).

## Tools vocabulary

- `none`: no external tools
- `code`: code interpreter / sandbox
- `web`: web search
- `file`: file system access
- `all`: all of the above
- `benchmark-specified`: tools required by the benchmark itself (e.g., OSWorld, SWE-bench, terminal-bench). Use this when the benchmark inherently requires tools — the model can't opt out.
