# Project notes — work log and outcomes

Single consolidated note for engineering decisions, completed work, and measured results. (Supersedes the old `CLEANUP_PLAN.md` checklist, which was review-only.)

---

## 1. Grounding / CCR (vector threshold)

**Done**

- `extras/experiments/french_medical_ner/compare_neo4j_grounding_ccr.py`: added `--vector-threshold` (float, default **0.6**), passed into `TermGraph(..., vector_score_threshold=...)`.
- `pipeline/graph.py`:
  - Default `vector_score_threshold` set to **0.6**.
  - **Vector mode:** if the vector index returns no rows, or the **top score &lt; threshold**, increment `vector_fallbacks` and return **`None`** (no silent string fallback).
  - **Vector + LLM:** only candidates with **score ≥ threshold** are passed to the LLM; if none qualify, same fallback counter and **`None`**.
- **Rationale:** Previously a low vector score could still yield a “grounded” span via string match, inflating CCR towards 1.0.

**Measured (example run, `segments_ner_biollm.jsonl`, 472 spans)**

| Mode | CCR | Vec FB (approx.) | Notes |
|------|-----|------------------|--------|
| string | ~0.35 | — | Lexical / fuzzy MedDRA FR match; ambiguous FR logged. |
| vector @ 0.6 | 1.00 | 0 | On this corpus every top-1 cosine was **≥ 0.6**; threshold still enforced (no string rescue). |
| vector_llm @ 0.6 | 1.00 | 0 | Same filtering before LLM. |
| vector only @ **0.85** | ~0.13 | ~410 | Illustrates threshold sensitivity (from a threshold sweep on the same segments). |

Stricter `--vector-threshold` (e.g. **0.85**) is useful when reporting “confidence-filtered” vector CCR.

---

## 2. `biomistral_prompt_ner.py` (GPU memory)

**Done:** After the HuggingFace and Unsloth backends finish writing the output JSONL, a **`finally`** block runs `del model`, `del tokenizer`, `gc.collect()`, and `torch.cuda.empty_cache()` (CUDA only). **Ollama** backend unchanged.

---

## 3. Dependencies (Unsloth)

**Done**

- `requirements.txt`: added **`unsloth_zoo`** and documented install order (`unsloth_zoo` → `trl` / `datasets` → `unsloth[cu124]` with `--break-system-packages` where needed).
- `extras/experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py`: `ImportError` text mentions `unsloth_zoo`.

**Caveat:** `unsloth_zoo` can pull a different **torch** stack than the rest of the project; keep one coherent CUDA line (see §5).

---

## 4. Environment / PyTorch (Fedora)

**Symptom:** `ImportError: libtorch_cuda.so: undefined symbol: ncclDevCommDestroy` when importing `torch` (often via `sentence_transformers` in `compare_neo4j_grounding_ccr`).

**Cause:** Mixed **CUDA 12** (`nvidia-*-cu12`) wheels alongside **torch 2.11 + CUDA 13** (`nvidia-nccl-cu13`, …) so the loader binds an older `libnccl`.

**Fix applied (conceptual)**

1. `pip uninstall` all **`nvidia-*-cu12`** packages still listed in the venv.
2. `pip install --force-reinstall nvidia-cudnn-cu13==9.19.0.56` (refreshes aligned `nvidia-cublas` / `nvidia-cuda-nvrtc`).

Then `import torch` and `compare_neo4j_grounding_ccr.py` succeeded (`torch 2.11.0+cu130` in the working snapshot).

---

## 5. Repository cleanup (this pass)

**Done**

- **Single PDF folder:** `data/test_data/` holds **`keytruda_fr.pdf`** and **`keytruda_en.pdf`** (copies of the former EPAR PDFs; not committed). **`Test_data/` removed** to avoid two trees; `tools/data/prepare_data.py` defaults point at `data/test_data/…` (see `data/test_data/README.md`).
- **Duplicate segment JSONL:** `segments_ner.jsonl` and `segments_ner_finetuned.jsonl` were **byte-identical**; **`segments_ner_finetuned.jsonl`** was a **symlink** to **`segments_ner.jsonl`** until the **5b** cleanup archived that symlink name under **`archive/data/section48/`** (generic **`segments_ner.jsonl`** remains).
- **Legacy evaluation figures:** older `results/figures/` snapshots were removed; use **`results/<condition>/figures/`** from `plot_figures.py`.
- **`rerun_all.sh`:** GPU pre-kill block shortened (single `TERM` pass + short sleep; optional `SKIP_GPU_KILL=1` unchanged).
- **Planning PDFs:** **`Project Plan/`** moved to **`docs/project-plan/`** (no space in path).
- **`CLEANUP_PLAN.md` removed**; this file is the living summary.
- **`.gitignore`:** `unsloth_compiled_cache/`.

**Not removed**

- **`results/ad_hoc/*.jsonl`**: retained outputs from pipeline runs that used the default `--results-dir results` (not byte-identical to `results/ner_biollm/` in checks).
- **`tools/**/*.py`** and **`extras/experiments/**/*.py`**: entrypoints remain intentional pipeline or supplementary stages.

---

## 5b. Second cleanup (two-condition repo)

**Done**

- **`archive/data/section48/`**: moved baseline CamemBERT, Llama3-, Mistral-instruct-tagged segment JSONLs, and the legacy `segments_ner_finetuned.jsonl` symlink name; added README.
- **`data/section48/vector_ccr_all_models.json`**: trimmed to BioLLM + Unsloth-full stats (rerun `extras/experiments/vector_grounding/build_vector_ccr_reports.py` to refresh).
- **`results/htm_vector_comparison/`**: removed stale committed plots/CSVs; `.gitkeep` only until `rerun_all.sh` or `tools/eval/compare_htm_vector_thresholds.py` regenerates them.
- **`extras/experiments/french_medical_ner/quaero_brat_reader.py`**: QUAERO BRAT helpers (`ID2LABEL`, `load_quaero_brat`) used by `biomistral_ner_finetune_unsloth.py` (standalone module; legacy CamemBERT trainer removed).
- **`error_analysis/legacy/`**: sample error CSV tagged with removed conditions.
- **`tools/admin/archive_results_snapshot.sh`**: optional tarball of `results/`, `error_analysis/`, `planning_locks.json`.

**Removed symlink**

- **`segments_ner_finetuned.jsonl`** had pointed at **`segments_ner.jsonl`**; the symlink path was archived; generic **`segments_ner.jsonl`** remains for `tools/data/prepare_data.py` workflows.

---

## 6. Cross-NER HTM bars and chrF–HTM trade-off axes (May 2026)

**Done**

- **`tools/eval/plot_cross_ner_dashboard.py`:** HTM panel y-limits use **min/max of the plotted values** plus padding (capped at 1.05), with the axis-break cue when the floor is above zero — avoids a full 0–1 scale when all HTM bars sit in a narrow band (~0.35–0.45).
- **`tools/eval/plot_figures.py`:** `scan_global_scatter_limits` now reads only **`ner_biollm`** and **`ner_biollm_finetuned`** `scores_summary.csv` files so legacy `ner_*` trees do not skew shared trade-off / bubble axis locks.
- **`plot_tradeoff` / `plot_bubble_chrF_htm_time`:** after merged limits are applied, **chrF** and **HTM** axes are tightened when the axis span is much wider than the point spread (keeps the “prefer top-right” shading but matches the visible cloud).

---

## 7. Session log PDF

**Added**

- [`docs/session_logs/TermPlanMT_Session_Log-5.pdf`](session_logs/TermPlanMT_Session_Log-5.pdf) — export of an interactive TermPlan-MT working session (plots, notes, debugging). Same bytes as the local download `~/Downloads/TermPlanMT_Session_Log-5.pdf` at the time of import; not a regulatory artefact.

---

## 8. Documentation map

| File | Purpose |
|------|---------|
| `README.md` | What the project is, reproduce steps, main results table. |
| `docs/REPO_LAYOUT.md` | Directory + script grouping (this layout pass). |
| `docs/PROJECT_NOTES.md` | This engineering log. |
| `docs/error_analysis/schema.md` | Manual QA schema. |
| `docs/session_logs/` | Optional PDF exports from interactive evaluation sessions (see §7). |
| `docs/project-plan/` | Proposal PDFs. |

---

## 9. Open items (not automated here)

- **`run_pipeline.py` / `evaluate.py`:** they construct `TermGraph` without CLI `--vector-threshold`; default in code is **0.6**. Override via future CLI/env for parity with `compare_neo4j_grounding_ccr` sweeps. *(Paths: `tools/pipeline/run_pipeline.py`, `tools/eval/evaluate.py`.)*
- **Unsloth fine-tune:** requires a GPU venv where `from unsloth import FastLanguageModel` succeeds; consider a **dedicated** env if `unsloth_zoo` and torch versions keep fighting.
