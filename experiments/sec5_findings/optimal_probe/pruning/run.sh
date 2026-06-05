#!/bin/bash
# Derive rank-pruned candidates from an existing all-known greedy result.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${SOURCE_GREEDY_RESULT:-}" ]]; then
  echo "SOURCE_GREEDY_RESULT is required; this runner never re-evaluates candidates." >&2
  exit 2
fi

args=(
  --source-greedy-result "$SOURCE_GREEDY_RESULT"
  --metric "${METRIC:-medae}"
)

if [[ -n "${KEEP_COUNT:-}" ]]; then
  args+=(--keep-count "$KEEP_COUNT")
else
  args+=(--keep-fraction "${KEEP_FRACTION:-0.30}")
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=(--max-steps "$MAX_STEPS")
fi

if [[ -n "${OUT:-}" ]]; then
  args+=(--out "$OUT")
fi

if [[ -n "${ALLOWLIST_OUT:-}" ]]; then
  args+=(--allowlist-out "$ALLOWLIST_OUT")
fi

python run.py "${args[@]}"
