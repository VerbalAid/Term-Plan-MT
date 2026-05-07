#!/usr/bin/env bash
# Full epoch-based ontology SFT (Qwen + LoRA under --fit-8gb). See extras/README.md.
# From repo root (or any cwd): bash extras/experiments/french_medical_ner/run_full_ontology_qwen_finetune.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec env PYTHONPATH=. "${ROOT}/.venv/bin/python" \
  extras/experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py \
  --fit-8gb \
  --ontology-only \
  --full-ontology-finetune \
  --resume-from-checkpoint auto \
  "$@"
