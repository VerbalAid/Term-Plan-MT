<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white" alt="Python 3.14+">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.x">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/sentence--transformers-embeddings-5A67D8?logo=python&logoColor=white" alt="sentence-transformers">
  <img src="https://img.shields.io/badge/Neo4j-graph%20store-008CC1?logo=neo4j&logoColor=white" alt="Neo4j">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker Compose">
  <img src="https://img.shields.io/badge/NumPy-%E2%89%A51.26-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-%E2%89%A53.8-11557c?logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/sacreBLEU-metrics-2F855A?logoColor=white" alt="sacreBLEU">
  <img src="https://img.shields.io/badge/pytest-tests-0A9EDC?logo=pytest&logoColor=white" alt="pytest">
</p>

# Terminology Aware Machine Translation

**TermPlan-MT** is a **French → English** translation framework for **SmPC Section 4.8**, built to enforce **MedDRA-grounded terminology consistency** across several machine translation setups.

It combines:

- **Named Entity Recognition (NER)** on French source text  
- **MedDRA grounding** via Neo4j (`TermGraph`)  
- **Global terminology planning** (per-surface locks shared across segments)  
- **Multiple MT systems** — plain and graph-aware — in `pipeline/systems/`  
- **Evaluation** that separates **fluency** (BLEU, chrF++, optional COMET) from **hierarchy-aware terminology** (**HTM**, **hyp–ref HTM agreement**, **rHTM**) and **dataset grounding coverage** (**CCR**), plus **document-level BLEU** variants (macro over sentences per document, and **concatenated** document BLEU in `scores_summary.csv`)

---

## 0. Thesis rerun in N steps

1. **Enter the repository** — From the parent folder, use a quoted path if your directory name ends with a **space** (some checkouts are literally `MT_Project_Terminology `):  
   `cd "/home/you/Desktop/Masters/MT/MT_Project_Terminology "`
2. **Create the venv and install dependencies** — `python -m venv .venv` then `.venv/bin/pip install -r requirements.txt` (see `requirements.txt` for optional Unsloth blocks used only from [`extras/`](extras/README.md)).
3. **Start Neo4j** — From the repo root: `docker compose up -d` (MedDRA graph build expectations are in [`data/README.md`](data/README.md)).
4. **Prepare or refresh segment JSONL** — Default segmentation: [`tools/data/prepare_data.py`](tools/data/prepare_data.py) → `data/section48/segments_ner.jsonl`. For BioMistral / Unsloth NER variants, run the scripts under [`extras/experiments/french_medical_ner/`](extras/experiments/french_medical_ner/) (see [`extras/README.md`](extras/README.md)).
5. **Run translation** — Full matrix: [`rerun_all.sh`](rerun_all.sh) from the repo root (`SKIP_*` toggles are documented in the script header). Single profile + lighter defaults: [`thesis_rerun.sh`](thesis_rerun.sh). Ad hoc: [`tools/pipeline/run_pipeline.py`](tools/pipeline/run_pipeline.py) with `--segments`, `--results-dir`, and optional `--system` / `--resume`.
6. **Evaluate and plot** — [`tools/eval/evaluate.py`](tools/eval/evaluate.py) and [`tools/eval/plot_figures.py`](tools/eval/plot_figures.py), or rely on Phase 2 inside `rerun_all.sh` / [`tools/eval/run_eval_plot_matrix.py`](tools/eval/run_eval_plot_matrix.py).

**S1 / S2 reuse (save GPU time):** In `rerun_all.sh`, `REUSE_S1_S2_FROM_BIOLLM=1` (default) copies `results/ner_biollm/s1.jsonl` and `s2.jsonl` into other result directories and runs **only S3–S5** for those conditions. Set `REUSE_S1_S2_FROM_BIOLLM=0` for a full S1–S5 rerun per directory. For one tree only, [`tools/pipeline/run_pipeline.py`](tools/pipeline/run_pipeline.py) supports `--resume` or `--system s3 s4 s5` when `s1.jsonl` / `s2.jsonl` are already complete.

**Shrinking the working tree:** Regenerated `results/` figures, `archive/`, and caches can be deleted locally when you only need **`tools/`** (see [`tools/README.md`](tools/README.md) for script locations); see [`.gitignore`](.gitignore) and [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md).

---

## 1. End-to-end pipeline overview

**Figure 1. High-level path from French SmPC segments to English hypotheses**

<p align="center">
  <img src="docs/figures/figure_pipeline_four_stage.png" alt="Four-stage pipeline overview" width="880"><br>
  <sub><b>Figure 1.</b> NER → grounding → planning → translation and metrics.</sub>
</p>

| Stage | Purpose |
| ----- | ------- |
| **NER** | Identify French medical spans worth constraining downstream (`terms[]` in JSONL) |
| **Grounding** | Map each span to a MedDRA concept in Neo4j (identifier, English label, hierarchy) |
| **Planning** | Choose a **canonical English rendering** per grounded surface and store it in a **global lock table** |
| **MT + metrics** | Run translators that consume `fr`, locks, and optional graph context; score fluency, **HTM** (hierarchy check on **English** `hyp`), and **CCR** |

---

## 2. Section 4.8 segmentation (test corpus)

The corpus is the **~10-page SmPC §4.8** slice (Keytruda PDFs under [`data/test_data/`](data/test_data/)), turned into **127 aligned sentence pairs** (default PDFs and alignment rule).

### Pipeline — [`tools/data/prepare_data.py`](tools/data/prepare_data.py)

1. **Extract §4.8** — Regex from the §4.8 heading through just before §4.9.  
2. **Sentence splitting** — French and English split independently with **NLTK Punkt**.  
3. **Alignment** — Pair sentences index-wise: `n = min(#FR sentences, #EN sentences)` → **127** segments.  
4. **JSONL** — One object per line, e.g.:

```json
{
  "id": "48_001",
  "fr": "...",
  "en_ref": "...",
  "terms": [...]
}
```

### Outputs

| Path | Role |
| ---- | ---- |
| [`data/section48/segments_ner.jsonl`](data/section48/segments_ner.jsonl) | Default segment file (CamemBERT-style `terms` when produced by this script) |
| `data/section48/segments_ner_*.jsonl` | NER variants: **same** `id` / `fr` / `en_ref`; only **`terms[]`** changes |

Re-run segmentation after any change to PDFs or tokenisation.

---

## 3. NER (Named Entity Recognition)

NER runs **before** grounding, planning, and translation.

### Output

Each segment carries a **`terms`** array (at minimum a French surface **`word`** per item; offsets and metadata depend on the extractor).

### Extractors and NER ablations (`extras/experiments/french_medical_ner/`)

GPU-heavy extractors and trainers live under **[`extras/`](extras/README.md)** (core `experiments/` is only a pointer — see [`experiments/README.md`](experiments/README.md)). The **primary evaluation** in `rerun_all.sh` contrasts **prompted** BioMistral NER vs **fine-tuned** BioMistral NER (`results/ner_biollm/` vs `results/ner_biollm_finetuned/`). Main scripts:

| Script | Purpose |
| ------ | ------- |
| [`biomistral_prompt_ner.py`](extras/experiments/french_medical_ner/biomistral_prompt_ner.py) | **Prompted** JSON-list extraction with **BioMistral-7B** (build or refresh `segments_ner_biollm.jsonl` from the aligned Section 4.8 JSONL) |
| [`biomistral_ner_finetune_unsloth.py`](extras/experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py) | **LoRA fine-tuning** with **Unsloth + TRL**; merged adapters load in **`biomistral_prompt_ner --backend unsloth`** |
| [`quaero_brat_reader.py`](extras/experiments/french_medical_ner/quaero_brat_reader.py) | **QUAERO** BRAT helpers (`ID2LABEL`, `load_quaero_brat`) for supervised NER training data |
| [`compare_neo4j_grounding_ccr.py`](extras/experiments/french_medical_ner/compare_neo4j_grounding_ccr.py) | **Grounding ablation:** string vs vector **CCR** over segment JSONLs; refuses vector modes if the Neo4j vector index is missing |

### Design rule

Downstream stages **do not re-run NER** — they only read **`terms[]`**.

---

## 4. Grounding (MedDRA / Neo4j)

Grounding maps a French string to a MedDRA concept using:

`TermGraph.ground(fr_term, context=sentence_fr)`

Typical payload includes MedDRA id, preferred English name, and hierarchy metadata. Implementation: [`pipeline/graph.py`](pipeline/graph.py).

### Modes (`grounding_mode`)

- **`string`** — Lexical / rule-based grounding in the graph  
- **`vector`** / **`vector_llm`** — Contextual embedding (and optional LLM assist) for disambiguation  

### Key threshold

- **`vector_score_threshold`** — Default **0.75** on `TermGraph`; some sweeps use **0.80**; keep training, planning, and eval configs aligned on purpose.

### Principle

If NER **never** extracts a term, grounding and planning **cannot** invent it later.

---

## 5. Planning (global terminology locks)

Implemented in [`pipeline/planning.py`](pipeline/planning.py).

### Goal

A **single global map** used by all segments and MT systems:

> French surface (from `terms[]`) → chosen English rendering for that concept

### Process (conceptual)

1. Walk segments and collect distinct French **`word`** values from **`terms[]`**.  
2. For each surface, call **`graph.ground(fr_term, context=that segment's French sentence)`**.  
3. Score English candidates (planning uses a **sentence-transformer** here; it is **not** the same mechanism as Neo4j vector grounding).  
4. Write decisions into a shared **lock** structure consumed by MT stages.

After translation, **HTM** ([§7](#7-htm-hierarchy-aware-terminology-metric)) applies the **same MedDRA hierarchy lens on the English hypothesis**—the reverse direction of grounding/planning, which attach concepts to **French** spans before decode.

### Artefact

Generated **`planning_locks.json`** typically lives next to the segment JSONL (e.g. [`data/section48/planning_locks.json`](data/section48/planning_locks.json) alongside `segments_ner*.jsonl`); regenerate when NER or segments change.

---

## 6. MT systems (translation stage)

Several backends under [`pipeline/systems/`](pipeline/systems/) consume:

- the French sentence **`fr`**  
- the **global lock table**  
- optional **graph-aware** context (per system design)

They write per-system **JSONL hypotheses** (`s1.jsonl` … `s5_mistral.jsonl`) aligned by segment **`id`**.

### Evaluation (after translation)

| Concern | What we measure |
| ------- | ----------------- |
| **Corpus BLEU** | **sacreBLEU** corpus BLEU over all aligned segment pairs: `BLEU().corpus_score(hypotheses, [references])` in [`pipeline/metrics/corpus_scores.py`](pipeline/metrics/corpus_scores.py) — one reference string per hypothesis, library defaults (effective order, tokenization). Shown as **`BLEU`** in `tools/eval/evaluate.py` and column **`bleu`** in `scores_summary.csv`. |
| **chrF++ / COMET** | Same alignment as BLEU: **chrF++** via sacreBLEU `CHRF().corpus_score`; **COMET** optional when dependencies and model weights are available (`tools/eval/evaluate.py`). |
| Document-level BLEU | **doc-BLEU** — unweighted macro average of corpus BLEU computed **per document** on segment lists (`bleu_doc_macro`; console **`BLdoc`**). **BLEU†** — same grouping, but each document is scored as **one** string by **joining** segment hypotheses and references with a space (`bleu_doc_concat` in `scores_summary.csv`; console **`BLcn`** in `evaluate.py`) |
| Terminology + hierarchy | **HTM** — MedDRA-aware check on **English hypotheses** (`htm.py`; NER `terms[]` + Neo4j grounding on the same `--segments` JSONL as CCR) |
| Hyp vs ref **ontology alignment** | **`htm_hyp_ref_agreement`** (per system): for each **grounded** French `terms[].word`, same HTM-style **1.0 / 0.5 / 0.0** score on `hyp` and on segment **`en_ref`**; mean of **1 − \|hyp_score − ref_score\|** (`compute_htm_hyp_vs_ref` in `htm.py`). Console column **`HypRefAg`** in `evaluate.py` (see stderr legend). **`--`** on the `(dataset)` row — not a dataset-level scalar. |
| Same HTM machinery vs **gold** `en_ref` | **rHTM** (dataset-level): how often grounded English renderings appear in the reference string (`compute_htm_en_ref` in `htm.py`; printed by `evaluate.py`, column **`htm_en_ref_dataset`** in `scores_summary.csv` from `plot_figures.py`) |
| How much NER is even grounded | **Dataset CCR** — fraction of extracted spans with non-null grounding (`pipeline/metrics/ccr.py`) |

---

## 7. HTM (hierarchy-aware terminology metric)

Implemented in [`pipeline/metrics/htm.py`](pipeline/metrics/htm.py).

### Conceptual role (mirror on English)

Upstream, **NER → grounding → planning** walks **French** text: find risky spans, attach **MedDRA** concepts, fix **English** renderings before decode. **HTM is the same idea run backwards on the model output:** it inspects the **English hypothesis `hyp`** and asks whether the terminology that actually appears there sits at the **right place in the MedDRA hierarchy** (same branch / level logic you use during grounding), instead of only scoring surface overlap with `en_ref`.

So the **object of analysis is always the translated English string**; Neo4j supplies the ontology constraints, just as in the pipeline.

### How it is implemented here

The runtime stack does **not** run a second full **English NER** pass over `hyp` in `htm.py`. **HTM** walks each segment’s French **`terms[].word`** (unique per segment), grounds the span in Neo4j with the segment French as context, builds allowed English renderings from the grounded concept, then scores **1.0 / 0.5 / 0.0** from what appears in **`hyp`** (`phrase_in_hyp`, `same_branch`, level match). The **same** segment JSONL you pass to **`--segments`** for CCR therefore defines which French spans are audited; the **score surface is always English in `hyp`**.

**rHTM** runs the **same** span-wise scoring but checks renderings against **`en_ref`** instead of `hyp`, aggregated once per dataset (useful as a lexical “MedDRA vs human reference” ceiling, not a per-system metric).

**Hypothesis–reference HTM agreement** (`compute_htm_hyp_vs_ref`): **per MT system**, for each French NER span that **grounds** in Neo4j, compute the usual HTM-style score on **`hyp`** and the same score on segment **`en_ref`**, then **1 − \|hyp_score − ref_score\|**; the reported value is the **mean** over those grounded spans (**`nan`** if none). **1.0** means the system’s ontology alignment matches the human reference segment for every audited span; **0.0** means maximum mismatch on that scale. It does **not** replace **HTM** or **rHTM**; it answers whether the model “lands” the same MedDRA rendering tier as the gold sentence for the same French cue.

### Scoring (per grounded NER span, simplified)

- **1.0** — Accepted English rendering **and** level consistent with the grounded concept  
- **0.5** — Related via **`same_branch`** in the graph but level check fails  
- **0.0** — Missing rendering, wrong branch, or no graph support  

Run-level **HTM** is the mean over scored spans. Variants: **`compute_htm`** (lexical), **`compute_htm_vector`** (embedding-assisted).

**Figure 2. HTM intuition**

<p align="center">
  <img src="docs/figures/figure_htm_metric.png" alt="HTM: hierarchy-aware terminology scoring example" width="820"><br>
  <sub><b>Figure 2.</b> After decode: English in <code>hyp</code> checked against MedDRA (same hierarchy discipline as upstream; audit points = French NER <code>terms[]</code> on the segment JSONL).</sub>
</p>

---

## 8. Repository navigation

| Location | Purpose |
| -------- | ------- |
| [`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md) | **Proposal-aligned conclusions:** SmPC scope, four stages, research questions, HTM/rHTM/CCR/fluency, contributions, limitations |
| [`docs/interpretation_of_results_snapshots.md`](docs/interpretation_of_results_snapshots.md) | **Main results narrative:** BioMistral prompt vs fine-tuned NER, CCR, cross-condition figures |
| [`docs/README.md`](docs/README.md) | Index of all documentation |
| [`tools/data/prepare_data.py`](tools/data/prepare_data.py) | Build §4.8 aligned JSONL (**127** segments for default Keytruda PDFs) |
| [`pipeline/`](pipeline/) | Graph, planning, MT systems, metrics |
| [`extras/README.md`](extras/README.md) | **Supplementary NER / training / vector grounding** (GPU-heavy scripts under `extras/experiments/`) |
| [`experiments/README.md`](experiments/README.md) | Pointer to `extras/` (legacy path kept for short citations) |
| [`docs/appendix_historical_ner_and_pipeline_results.md`](docs/appendix_historical_ner_and_pipeline_results.md) | Historical NER conditions and metric tables |
| [`data/README.md`](data/README.md) | Data layout, optional graph seed JSON, MedDRA folder expectations |
| [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md) | Full tree-oriented map of the repo |

---

## 9. MedDRA licensing

MedDRA is **not** redistributed in this repository. Obtain a licence and data from [meddra.org/software-packages](https://www.meddra.org/software-packages), then load into Neo4j per your internal graph build (`tools/data/extract_meddra.py`, `tools/data/build_graph.py` — see `data/README.md`).

---

## Full narrative

High-level conclusions (problem, research questions, stages, metrics): [`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md).

Methodology and evaluation plan (when present): [`docs/project-plan/TermPlanMT_Proposal.pdf`](docs/project-plan/TermPlanMT_Proposal.pdf).
