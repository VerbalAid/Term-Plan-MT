#!/usr/bin/env bash
# Reproduce: BioMistral-prompt NER + fine-tuned BioMistral NER only.
# Excludes segment 48_028 (Section 4.8 Tableau 2 / dense table block) from translation and metrics by default.
#
# Override exclusions: EXCLUDE_SEGMENT_IDS=""  (include all segments)
# Skip phases: SKIP_GPU_KILL=1 SKIP_NER_BIOLLM=1 SKIP_NER_BIOMISTRAL_FT=1 SKIP_EVAL_PHASE=1
#               SKIP_EVAL_MATRIX=1 (skip tools/eval/run_eval_plot_matrix.py entirely)
#               SKIP_EVAL_NER_BIOLLM=1 / SKIP_EVAL_NER_BIOLLM_FT=1 (skip that profile only; other EVAL_RERUN_PROFILES still run)
#               SKIP_CROSS_NER_DASHBOARD=1 SKIP_HTM_THRESHOLD_COMPARE=1 REUSE_S1_S2_FROM_BIOLLM=0 EXTRA_RUN_PIPELINE_FLAGS="--resume"
# HTM in evaluate.py + plot_figures.py: uses NER ``terms[]`` from the same ``--segments`` JSONL as CCR.
# Optional vector HTM columns (separate from lexical ``htm``): HTM_VECTOR_THRESHOLDS=0.8,0.9
# HTM threshold script: runs when SKIP_HTM_THRESHOLD_COMPARE is unset; optional EXTRA_HTM_COMPARE_FLAGS.
#
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
  echo "nvidia-smi not found; skipping GPU kill (CPU-only or driver not in PATH)."
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Expected venv at ${PYTHON}; create it and pip install -r requirements.txt" >&2
  exit 1
fi

FLAGS="${EXTRA_RUN_PIPELINE_FLAGS:-${EXTRA_RUN_SYSTEMS_FLAGS:-}}"

run_eval_plot_matrix() {
  echo "========================================================================"
  echo "EVAL + FIGURES — pipeline/metrics/eval_manifest.EVAL_RERUN_PROFILES"
  echo "  (skip dirs with no pipeline JSONLs or missing segment files)"
  echo "========================================================================"
  export EVAL_GROUNDING_MODES="${EVAL_GROUNDING_MODES:-string}"
  export HTM_VECTOR_THRESHOLDS="${HTM_VECTOR_THRESHOLDS:-}"
  export EXTRA_EVAL_FLAGS="${EXTRA_EVAL_FLAGS:-}"
  export PLOT_COMET="${PLOT_COMET:-0}"
  local _skip_prof="${SKIP_EVAL_PROFILES:-}"
  if [[ "${SKIP_EVAL_NER_BIOLLM:-0}" == "1" ]]; then
    _skip_prof="${_skip_prof},ner_biollm"
  fi
  if [[ "${SKIP_EVAL_NER_BIOLLM_FT:-0}" == "1" ]]; then
    _skip_prof="${_skip_prof},ner_biollm_finetuned"
  fi
  _skip_prof="${_skip_prof#,}"
  _skip_prof="${_skip_prof%,}"
  export SKIP_EVAL_PROFILES="${_skip_prof}"
  # shellcheck disable=SC2086
  "${PYTHON}" tools/eval/run_eval_plot_matrix.py --exclude-segment-ids "${EXCLUDE_IDS}"
}

run_pipeline_only() {
  local title="$1"
  local segments_rel="$2"
  local results_sub="$3"
  local skip_env="${4:-}"

  local seg="${ROOT}/${segments_rel}"
  local rd="${ROOT}/${results_sub}"

  echo "========================================================================"
  echo "PIPELINE — ${title}"
  echo "  segments: ${segments_rel}"
  echo "  exclude:  ${EXCLUDE_IDS:-∅}"
  echo "  outputs:  ${results_sub}/"
  echo "========================================================================"

  if [[ ! -f "${seg}" ]]; then
    echo "SKIP — segments file missing: ${seg}"
    return 0
  fi

  if [[ -n "${skip_env}" ]]; then
    local _vn="${skip_env}"
    if [[ "${!_vn:-0}" == "1" ]]; then
      echo "SKIP pipeline (${skip_env}=1) — using existing JSONL under ${results_sub}/"
      return 0
    fi
  fi

  local pipe_sys=(--system all)
  if [[ "${REUSE_S1_S2_FROM_BIOLLM:-1}" == "1" ]] && [[ "${results_sub}" != "results/ner_biollm" ]]; then
    local donor="${ROOT}/results/ner_biollm"
    if [[ -f "${donor}/s1.jsonl" ]] && [[ -f "${donor}/s2.jsonl" ]]; then
      echo "Reusing ${donor}/s1.jsonl + s2.jsonl → ${results_sub}/ (pipeline runs s3–s5 only)."
      mkdir -p "${rd}"
      cp "${donor}/s1.jsonl" "${rd}/s1.jsonl"
      cp "${donor}/s2.jsonl" "${rd}/s2.jsonl"
      pipe_sys=(--system s3 s4 s5)
    else
      echo "WARN: ${donor}/s1.jsonl or s2.jsonl missing — running full s1–s5 for this condition."
    fi
  fi

  # shellcheck disable=SC2086
  "${PYTHON}" tools/pipeline/run_pipeline.py "${pipe_sys[@]}" --s5-backend both \
    --results-dir "${results_sub}" \
    --segments "${segments_rel}" \
    --exclude-segment-ids "${EXCLUDE_IDS}" \
    ${FLAGS}
}

echo ""
echo "################################################################################"
echo "# PHASE 1 — PIPELINES (BioMistral prompt + FT BioMistral NER; exclude ids: ${EXCLUDE_IDS:-none})"
echo "################################################################################"
echo ""

run_pipeline_only \
  "NER — BioMistral-7B JSON-list prompting" \
  "data/section48/segments_ner_biollm.jsonl" \
  "results/ner_biollm" \
  "SKIP_NER_BIOLLM"

if [[ "${SKIP_NER_BIOMISTRAL_FT:-0}" != "1" ]]; then
  # Fourth arg (skip env) omitted on purpose: empty skip_env; avoids a bare "" line some bash parses badly.
  run_pipeline_only \
    "NER - Fine-tuned BioMistral NER (Unsloth merged / LoRA)" \
    "${UNSLOTH_SEG}" \
    "results/ner_biollm_finetuned"
else
  echo "SKIP FT BioMistral NER pipeline (SKIP_NER_BIOMISTRAL_FT=1)"
fi

if [[ "${SKIP_EVAL_PHASE:-0}" == "1" ]]; then
  echo ""
  echo "SKIP_EVAL_PHASE=1 — skipping metrics/plots, cross-NER dashboard, and HTM threshold comparison."
  echo "Done (pipelines only)."
  exit 0
fi

echo ""
echo "################################################################################"
echo "# PHASE 2 — EVALUATION + FIGURES (eval_manifest.EVAL_RERUN_PROFILES)"
echo "################################################################################"
echo ""

if [[ "${SKIP_EVAL_MATRIX:-0}" == "1" ]]; then
  echo "SKIP_EVAL_MATRIX=1 — skipping tools/eval/run_eval_plot_matrix.py"
elif [[ "${SKIP_EVAL_NER_BIOLLM:-0}" == "1" ]] && [[ "${SKIP_EVAL_NER_BIOLLM_FT:-0}" == "1" ]]; then
  echo "SKIP_EVAL_NER_BIOLLM=1 and SKIP_EVAL_NER_BIOLLM_FT=1 — skipping run_eval_plot_matrix (matches legacy: no eval for the two BioLLM dirs)."
  echo "  (Cross-NER dashboard + HTM threshold compare still run below; use SKIP_EVAL_MATRIX=1 only if you meant skip matrix explicitly.)"
else
  run_eval_plot_matrix
fi

if [[ "${SKIP_CROSS_NER_DASHBOARD:-0}" != "1" ]]; then
  echo "========================================================================"
  echo "Cross-NER dashboard (conditions from eval_manifest.EVAL_RERUN_PROFILES with scores_summary)"
  echo "========================================================================"
  "${PYTHON}" tools/eval/plot_cross_ner_dashboard.py \
    --results-root "${ROOT}/results" \
    --figures-subdir "${CROSS_FIGURES_SUBDIR:-figures}" \
    --out-dir "${ROOT}/results/cross_ner_comparison" \
    --exclude-segment-ids "${EXCLUDE_IDS}"
fi

if [[ "${SKIP_HTM_THRESHOLD_COMPARE:-0}" != "1" ]]; then
  echo "========================================================================"
  echo "HTM string vs vector thresholds (Neo4j + sentence-transformers)"
  echo "========================================================================"
  # shellcheck disable=SC2086
  "${PYTHON}" tools/eval/compare_htm_vector_thresholds.py \
    --results-root "${ROOT}/results" \
    --exclude-segment-ids "${EXCLUDE_IDS}" \
    ${EXTRA_HTM_COMPARE_FLAGS:-}
fi

echo "Done."
echo "  BioMistral prompt:     ${ROOT}/results/ner_biollm/figures/"
echo "  FT BioMistral NER:     ${ROOT}/results/ner_biollm_finetuned/figures/ (segments: ${UNSLOTH_SEG})"
echo "  Cross-NER figures:     ${ROOT}/results/cross_ner_comparison/"
echo "  HTM threshold plots:   ${ROOT}/results/htm_vector_comparison/ (skip: SKIP_HTM_THRESHOLD_COMPARE=1)"
echo "  Default excluded ids:  ${EXCLUDE_IDS} (segment 48_028 = Tableau 2 block); set EXCLUDE_SEGMENT_IDS= to include all segments."
