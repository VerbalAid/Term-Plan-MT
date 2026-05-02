#!/usr/bin/env bash
# One-shot tarball of paper-facing outputs + planning locks (optional segment sources).
# Usage: ./scripts/archive_results_snapshot.sh [extra-path ...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
stamp="$(date +%Y%m%d_%H%M%S)"
out="artifacts_snapshot_${stamp}.tar.gz"
paths=(results exports data/section48/planning_locks.json)
for extra in "$@"; do
  paths+=("$extra")
done
tar -czvf "$out" "${paths[@]}"
echo "Wrote ${ROOT}/${out}"
