#!/usr/bin/env bash
# One-shot tarball of paper-facing outputs + planning locks (optional segment sources).
# Usage: ./tools/admin/archive_results_snapshot.sh [extra-path ...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
stamp="$(date +%Y%m%d_%H%M%S)"
out="artifacts_snapshot_${stamp}.tar.gz"
paths=(results error_analysis data/section48/planning_locks.json)
if [[ -d archive/logs ]] && [[ -n "$(ls -A archive/logs 2>/dev/null)" ]]; then
  paths+=("archive/logs")
fi
for extra in "$@"; do
  paths+=("$extra")
done
tar -czvf "$out" "${paths[@]}"
echo "Wrote ${ROOT}/${out}"
