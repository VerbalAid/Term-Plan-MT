#!/usr/bin/env bash
# Reproduce paper results: BioMistral-prompt NER and fine-tuned BioMistral NER conditions.
# Segment 48_028 (Section 4.8 Table 2 block) is excluded by default.
#
# Skip flags:
#   EXCLUDE_SEGMENT_IDS=""         include all segments
#   SKIP_GPU_KILL=1               do not kill GPU processes before starting
#   SKIP_NER_BIOLLM=1             skip pipeline for ner_biollm condition
#   SKIP_NER_BIOMISTRAL_FT=1      skip pipeline for ner_biollm_finetuned condition
#   SKIP_EVAL_PHASE=1             skip all eval/plots
#   SKIP_EVAL_MATRIX=1            skip run_eval_plot_matrix.py
#   SKIP_EVAL_NER_BIOLLM=1        skip eval for ner_biollm only
#   SKIP_EVAL_NER_BIOLLM_FT=1     skip eval for ner_biollm_finetuned only
#   SKIP_CROSS_NER_DASHBOARD=1    skip cross-condition comparison plots
#   REUSE_S1_S2_FROM_BIOLLM=0     re-run S1/S2 for finetuned condition (default: copy from biollm)
#   EXTRA_RUN_PIPELINE_FLAGS="--resume"  append extra flags to run_pipeline.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

UNSLOTH_SEG="data/section48/segments_ner_unsloth.jsonl"
if [[ ! -f "${ROOT}/${UNSLOTH_SEG}" ]]; then
  UNSLOTH_SEG="data/section48/segments_ner_unsloth_full.jsonl"
fi

EXCLUDE_IDS="${EXCLUDE_SEGMENT_IDS:-48_028}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${SKIP_GPU_KILL:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  _pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)"
  if [[ -n "${_pids}" ]]; then
    echo "Stopping GPU compute PIDs: ${_pids//$'\n'/ }"
    printf '%s\n' "${_pids}" | while IFS= read -r _pid; do
      [[ -z "${_pid}" ]] && continue
      kill -TERM "${_pid}" 2>/dev/null || true
    done
    sleep 2
  else
    echo "No GPU compute processes listed by nvidia-smi."
  fi
elif [[ "${SKIP_GPU_KILL:-0}" != "1" ]]; then
  echo "nvidia-smi not found; skipping GPU kill."
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Expected venv at ${PYTHON}; run: python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

FLAGS="${EXTRA_RUN_PIPELINE_FLAGS:-}"

# ── Helper: run pipeline for one NER condition ─────────────────────────────

run_pipeline_only() {
  local title="$1" segments_rel="$2" results_sub="$3" skip_env="${4:-}"
  local seg="${ROOT}/${segments_rel}" rd="${ROOT}/${results_sub}"

  echo "========================================================================"
  echo "PIPELINE — ${title}"
  echo "  segments: ${segments_rel}  exclude: ${EXCLUDE_IDS:-∅}"
  echo "========================================================================"

  if [[ ! -f "${seg}" ]]; then
    echo "SKIP — segments file missing: ${seg}"
    return 0
  fi

  if [[ -n "${skip_env}" ]] && [[ "${!skip_env:-0}" == "1" ]]; then
    echo "SKIP pipeline (${skip_env}=1)"
    return 0
  fi

  local pipe_sys=(--system all)
  if [[ "${REUSE_S1_S2_FROM_BIOLLM:-1}" == "1" ]] && [[ "${results_sub}" != "results/ner_biollm" ]]; then
    local donor="${ROOT}/results/ner_biollm"
    if [[ -f "${donor}/s1.jsonl" ]] && [[ -f "${donor}/s2.jsonl" ]]; then
      echo "Reusing ${donor}/s1.jsonl + s2.jsonl (running S3–S5 only)."
      mkdir -p "${rd}"
      cp "${donor}/s1.jsonl" "${rd}/s1.jsonl"
      cp "${donor}/s2.jsonl" "${rd}/s2.jsonl"
      pipe_sys=(--system s3 s4 s5)
    else
      echo "WARN: s1/s2 not found in ner_biollm — running full S1–S5."
    fi
  fi

  # shellcheck disable=SC2086
  "${PYTHON}" run_pipeline.py "${pipe_sys[@]}" --s5-backend both \
    --results-dir "${results_sub}" \
    --segments "${segments_rel}" \
    --exclude-segment-ids "${EXCLUDE_IDS}" \
    ${FLAGS}
}

# ── Phase 1: pipelines ──────────────────────────────────────────────────────

echo ""
echo "########################################################################"
echo "# PHASE 1 — PIPELINES"
echo "########################################################################"
echo ""

run_pipeline_only \
  "ner_biollm (BioMistral-7B prompt)" \
  "data/section48/segments_ner_biollm.jsonl" \
  "results/ner_biollm" \
  "SKIP_NER_BIOLLM"

if [[ "${SKIP_NER_BIOMISTRAL_FT:-0}" != "1" ]]; then
  run_pipeline_only \
    "ner_biollm_finetuned (fine-tuned BioMistral NER)" \
    "${UNSLOTH_SEG}" \
    "results/ner_biollm_finetuned"
else
  echo "SKIP ner_biollm_finetuned pipeline (SKIP_NER_BIOMISTRAL_FT=1)"
fi

if [[ "${SKIP_EVAL_PHASE:-0}" == "1" ]]; then
  echo "SKIP_EVAL_PHASE=1 — done (pipelines only)."
  exit 0
fi

# ── Phase 2: evaluation + figures ──────────────────────────────────────────

echo ""
echo "########################################################################"
echo "# PHASE 2 — EVALUATION + FIGURES"
echo "########################################################################"
echo ""

if [[ "${SKIP_EVAL_MATRIX:-0}" != "1" ]]; then
  export EVAL_GROUNDING_MODES="${EVAL_GROUNDING_MODES:-string}"
  export HTM_VECTOR_THRESHOLDS="${HTM_VECTOR_THRESHOLDS:-}"
  export EXTRA_EVAL_FLAGS="${EXTRA_EVAL_FLAGS:-}"
  export PLOT_COMET="${PLOT_COMET:-0}"

  # Allow per-condition skipping via SKIP_EVAL_NER_BIOLLM / SKIP_EVAL_NER_BIOLLM_FT.
  _skip_prof="${SKIP_EVAL_PROFILES:-}"
  [[ "${SKIP_EVAL_NER_BIOLLM:-0}"    == "1" ]] && _skip_prof="${_skip_prof},ner_biollm"
  [[ "${SKIP_EVAL_NER_BIOLLM_FT:-0}" == "1" ]] && _skip_prof="${_skip_prof},ner_biollm_finetuned"
  _skip_prof="${_skip_prof#,}"; _skip_prof="${_skip_prof%,}"
  export SKIP_EVAL_PROFILES="${_skip_prof}"

  "${PYTHON}" run_eval_plot_matrix.py --exclude-segment-ids "${EXCLUDE_IDS}"
else
  echo "SKIP_EVAL_MATRIX=1 — skipping run_eval_plot_matrix.py"
fi

if [[ "${SKIP_CROSS_NER_DASHBOARD:-0}" != "1" ]]; then
  echo "========================================================================"
  echo "Cross-NER dashboard"
  echo "========================================================================"
  "${PYTHON}" plot_cross_ner_dashboard.py \
    --results-root "${ROOT}/results" \
    --figures-subdir "${CROSS_FIGURES_SUBDIR:-figures}" \
    --out-dir "${ROOT}/results/cross_ner_comparison" \
    --exclude-segment-ids "${EXCLUDE_IDS}"
fi

echo ""
echo "Done."
echo "  ner_biollm figures:           results/ner_biollm/figures/"
echo "  ner_biollm_finetuned figures: results/ner_biollm_finetuned/figures/"
echo "  Cross-NER figures:            results/cross_ner_comparison/"
echo "  Excluded segment ids:         ${EXCLUDE_IDS:-none} (set EXCLUDE_SEGMENT_IDS= to include all)"
