#!/usr/bin/env bash
# Run NER with a saved HF Trainer LoRA checkpoint, then print dataset CCR (Neo4j grounding of NER spans).
# Usage (from repo root) — all paths required (no baked-in checkpoint):
#   bash training_scripts/ner/run_ner_ccr_from_lora_checkpoint.sh \\
#     data/section48/segments_ner_my_run.jsonl models/my-lora/checkpoint-5000 unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit
#
# Arg 3 (HF base) must match ``base_model_name_or_path`` in the checkpoint's adapter_config.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <segments_out.jsonl> <lora_checkpoint_dir> [hf_base_model_id]" >&2
  echo "example: $0 data/section48/segments_ner_x.jsonl models/my-qwen-lora/checkpoint-5000 unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit" >&2
  exit 1
fi

SEG_OUT="${1:?}"
LORA_CKPT="${2:?}"
BASE="${3:-unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit}"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv: ${PY}" >&2
  exit 1
fi

echo "== NER (Unsloth base + LoRA checkpoint) =="
echo "  base:  ${BASE}"
echo "  lora:  ${LORA_CKPT}"
echo "  out:   ${SEG_OUT}"
# shellcheck disable=SC2086
PYTHONPATH=. "${PY}" training_scripts/ner/biomistral_prompt_ner.py \
  --backend unsloth \
  --unsloth-base-model "${BASE}" \
  --unsloth-lora-path "${LORA_CKPT}" \
  --output "${SEG_OUT}"

echo ""
echo "== CCR (dataset — NER spans vs Neo4j / MedDRA graph; needs Docker Neo4j) =="
# shellcheck disable=SC2086
PYTHONPATH=. "${PY}" tools/eval/evaluate.py \
  --ccr-only \
  --segments "${SEG_OUT}" \
  --grounding-mode string

echo ""
echo "Optional: full pipeline + tables, e.g."
echo "  PYTHONPATH=. ${PY} tools/pipeline/run_pipeline.py --segments ${SEG_OUT} --results-dir results/ner_qwen25_meddra_ckpt5000"
echo "  PYTHONPATH=. ${PY} tools/eval/evaluate.py --segments ${SEG_OUT} --results-dir results/ner_qwen25_meddra_ckpt5000"
