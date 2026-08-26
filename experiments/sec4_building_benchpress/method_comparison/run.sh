#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run.py"

START=0
END=328
LIMIT=""
WORKERS=12
FORCE=false
MERGE=false
PLOT=false
TABLE_OUT=""

usage() {
  cat <<'EOF'
Usage: run.sh [options]

Run Table 4 transform-method shards in parallel.

Options:
  --start N      First shard index (default: 0)
  --end N        Last shard index, inclusive (default: 328)
  --limit N      Run at most N shard indices from --start
  --workers N    Parallel Python processes (default: 12)
  --force        Recompute selected shards even when valid outputs exist
  --merge        Merge completed shards after the sweep
  --plot         Render the nested-selection plot after merging
  --table-out P  Write the generated Table 4 LaTeX to P (requires --merge)
  -h, --help     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --merge) MERGE=true; shift ;;
    --plot) PLOT=true; shift ;;
    --table-out) TABLE_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "${START}" "${END}" "${WORKERS}" ${LIMIT:+"${LIMIT}"}; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "Shard bounds, limit, and workers must be non-negative integers." >&2
    exit 2
  }
done

(( START <= END )) || {
  echo "--start must not exceed --end." >&2
  exit 2
}
(( END <= 328 )) || {
  echo "--end must not exceed 328." >&2
  exit 2
}
(( WORKERS > 0 )) || {
  echo "--workers must be positive." >&2
  exit 2
}
if [[ -n "${TABLE_OUT}" && "${MERGE}" != true ]]; then
  echo "--table-out requires --merge." >&2
  exit 2
fi
if [[ "${PLOT}" == true && "${MERGE}" != true ]]; then
  echo "--plot requires --merge." >&2
  exit 2
fi

if [[ -n "${LIMIT}" ]]; then
  (( LIMIT > 0 )) || {
    echo "--limit must be positive." >&2
    exit 2
  }
  LAST=$((START + LIMIT - 1))
  (( LAST < END )) && END="${LAST}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

FORCE_ARG=()
[[ "${FORCE}" == true ]] && FORCE_ARG=(--force)

seq "${START}" "${END}" |
  xargs -P "${WORKERS}" -I{} bash -c \
    'python "$1" --shard-index "$2" "${@:3}"' \
    _ "${RUNNER}" "{}" "${FORCE_ARG[@]}"

if [[ "${MERGE}" == true ]]; then
  python "${RUNNER}" --merge
  if [[ -n "${TABLE_OUT}" ]]; then
    mkdir -p "$(dirname "${TABLE_OUT}")"
    python "${SCRIPT_DIR}/gen_table.py" > "${TABLE_OUT}"
  fi
  if [[ "${PLOT}" == true ]]; then
    python "${SCRIPT_DIR}/plot.py"
  fi
fi
