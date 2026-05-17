# BenchPress Score Matrix (`benchpress/data/llm_benchmark_data.json`)

Sparse model × benchmark score matrix for BenchPress experiments.
Schema: `models[]`, `benchmarks[]`, `scores[{model_id, benchmark_id, score, reference_url}]`.

`benchmark_cost_evidence.json` is a separate evidence file for public,
mechanically extractable benchmark cost signals such as prompt/completion token
tables. It is not part of the score matrix schema; see
`benchmark_cost_evidence.README.md`.

Benchmark-level `cost` dictionaries inside `llm_benchmark_data.json` store the
current normalized token/dollar evidence attached to each benchmark. They should
point back to a raw source in `benchmark_cost_evidence.json` when possible.

## Score collection conventions

When a source reports multiple numbers for the same (model, benchmark), use this
preference order — top wins:

1. **No tools / no search / no code execution**
   When a vendor or leaderboard distinguishes "no tools" vs "tool-use / code-exec /
   search-augmented" numbers, **always pick the no-tools number**. BenchPress
   measures raw model capability, not agentic harness performance.
   - Example: AIME 2025, Gemini 3 Flash → `95.2` (no tools), not `99.7` (with code execution).
   - Example: SWE-bench Verified, GPT-5.2 → use the bare-model number, not the agent-with-Codex score.

2. **Reasoning / thinking ON** when the model has a default reasoning mode that
   the vendor highlights as the headline number (Gemini 3 Pro Thinking, Claude
   Sonnet 4.5 Thinking, etc.). Pick the "Thinking" / "Extra-high reasoning" /
   "high effort" headline variant — it's what the vendor reports as the model's
   official benchmark.

3. **Single-attempt** (pass@1, maj@1) over self-consistency / pass@k aggregations.

## Source URL preference order

Pick the most authoritative URL that actually contains the number you recorded:

1. **Vendor model page** — e.g. `deepmind.google/models/gemini/flash/`,
   `anthropic.com/news/claude-...`, `openai.com/index/...`.
   Highest authority for vendor-reported numbers.

2. **Official benchmark leaderboard** — e.g. `arcprize.org/arc-agi/2/`,
   `swebench.com`, `epoch.ai/benchmarks/frontiermath`,
   `matharena.ai/competition_tables/<series>--<comp_id>`,
   `lmarena.ai`, `tbench.ai/leaderboard/...`.
   Use when the vendor page doesn't list the number.

3. **Aggregator** — `vellum.ai/blog/...`, `artificialanalysis.ai`,
   `livebench.ai`. Use only as last resort. Aggregators sometimes lag or
   misattribute numbers.

4. **Avoid** — Medium posts, Reddit, screenshots, "preliminary review" blogs.

## URL gotchas

- **matharena.ai**: the `/` root is JS-rendered. Use the data-endpoint pattern
  `https://matharena.ai/competition_tables/<series>--<comp_id>`. Examples:
  - `aime--aime_2025`, `aime--aime_2024`
  - `hmmt--hmmt_feb_2025`, `hmmt--hmmt_nov_2025`
  - `brumo--brumo_2025`, `cmimc--cmimc_2025`, `smt--smt_2025`
  - `matharena_apex--matharena_apex_2025`

- **Codeforces ratings**: many "rating" numbers appear in vendor blog posts, not
  on Codeforces itself. Cite the vendor source.

- **MMLU vs MMLU-Pro vs MMMLU**: these are three separate benchmarks. Don't
  conflate. MMMLU is the multilingual MMLU; MMLU-Pro is the harder reformulation.

## Audit & fix workflow

See `../../others/score_audit_menu.md` for the active checklist of
(model, benchmark) pairs flagged by the score audit. Fix loop:

1. Pick a `[ ]` row from the menu.
2. Verify against the highest-priority source per the rules above.
3. Update `score` and `reference_url` in this JSON.
4. Tick `[x]` in the menu and note the new value + source.
