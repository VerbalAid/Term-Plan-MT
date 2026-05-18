#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
exec python -m uvicorn webapp.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
