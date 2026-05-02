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
- **Evaluation** that separates **fluency** (BLEU, chrF++, optional COMET) from **hierarchy-aware terminology** (**HTM**) and **dataset grounding coverage** (**CCR**)

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
| **MT + metrics** | Run translators that consume `fr`, locks, and optional graph context; score fluency, **HTM**, and **CCR** |

---

## 2. Section 4.8 segmentation (test corpus)

The corpus is the **~10-page SmPC §4.8** slice (Keytruda PDFs under [`test_data/`](test_data/)), turned into **127 aligned sentence pairs** (default PDFs and alignment rule).

### Pipeline — [`scripts/prepare_data.py`](scripts/prepare_data.py)

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

### Extractors and NER ablations (`experiments/french_medical_ner/`)

This folder holds **French medical span extraction** for SmPC §4.8, **fine-tuning** utilities, and **ablation / diagnostic** scripts (for example CCR under different Neo4j grounding modes). The **primary evaluation** in `rerun_all.sh` contrasts **prompted** BioMistral NER vs **fine-tuned** BioMistral NER (`results/ner_biollm/` vs `results/ner_biollm_finetuned/`); the table below lists the main entry points.

| Script | Purpose |
| ------ | ------- |
| [`biomistral_prompt_ner.py`](experiments/french_medical_ner/biomistral_prompt_ner.py) | **Prompted** JSON-list extraction with **BioMistral-7B** (build or refresh `segments_ner_biollm.jsonl` from the aligned Section 4.8 JSONL) |
| [`biomistral_ner_finetune_unsloth.py`](experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py) | **LoRA fine-tuning** with **Unsloth + TRL**; merged adapters load in **`biomistral_prompt_ner --backend unsloth`** |
| [`quaero_brat_reader.py`](experiments/french_medical_ner/quaero_brat_reader.py) | **QUAERO** BRAT helpers (`ID2LABEL`, `load_quaero_brat`) for supervised NER training data |
| [`compare_neo4j_grounding_ccr.py`](experiments/french_medical_ner/compare_neo4j_grounding_ccr.py) | **Grounding ablation:** string vs vector **CCR** over segment JSONLs; refuses vector modes if the Neo4j vector index is missing |

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
| Fluency vs reference | **BLEU**, **chrF++**, optional **COMET** (`scripts/evaluate.py`, `scripts/plot_results.py`) |
| Terminology + hierarchy | **HTM** via optional `--gold-terms` JSON + Neo4j (`pipeline/metrics/htm.py`) |
| How much NER is even grounded | **Dataset CCR** — fraction of extracted spans with non-null grounding (`pipeline/metrics/ccr.py`) |

---

## 7. HTM (hierarchy-aware terminology metric)

Implemented in [`pipeline/metrics/htm.py`](pipeline/metrics/htm.py).

### Purpose

Score whether English in the hypothesis respects **gold-listed** terminology **and** MedDRA **branch / level** consistency when a gold French cue appears in the source.

HTM is **not** a free-running measure of “specificity drift” from French to English without a reference list: it always uses **curated gold rows** (French substring in `fr`, expected English renderings, level) plus **Neo4j** to judge the hypothesis.

### Scoring (per gold hit, simplified)

- **1.0** — Accepted English rendering **and** level consistent with the gold row  
- **0.5** — Related via **`same_branch`** in the graph but level check fails  
- **0.0** — Missing rendering, wrong branch, or no graph support  

Run-level **HTM** is the mean over scored terms. Variants: **`compute_htm`** (lexical), **`compute_htm_vector`** (embedding-assisted).

**Figure 2. HTM intuition**

<p align="center">
  <img src="docs/figures/figure_htm_metric.png" alt="HTM: hierarchy-aware terminology scoring example" width="820"><br>
  <sub><b>Figure 2.</b> Gold French cue → grounded English choices in Neo4j (see <code>htm.py</code>).</sub>
</p>

---

## 8. Repository navigation

| Location | Purpose |
| -------- | ------- |
| [`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md) | **Proposal-aligned conclusions:** SmPC scope, four stages, research questions, HTM/CCR/fluency, contributions, limitations |
| [`docs/interpretation_of_results_snapshots.md`](docs/interpretation_of_results_snapshots.md) | **Main results narrative:** BioMistral prompt vs fine-tuned NER, CCR, cross-condition figures |
| [`docs/README.md`](docs/README.md) | Index of all documentation |
| [`scripts/prepare_data.py`](scripts/prepare_data.py) | Build §4.8 aligned JSONL (**127** segments for default Keytruda PDFs) |
| [`pipeline/`](pipeline/) | Graph, planning, MT systems, metrics |
| [`experiments/french_medical_ner/`](experiments/french_medical_ner/) | French medical NER, fine-tuning, QUAERO I/O, grounding / CCR ablations |
| [`docs/appendix_historical_ner_and_pipeline_results.md`](docs/appendix_historical_ner_and_pipeline_results.md) | Historical NER conditions and metric tables |
| [`data/README.md`](data/README.md) | Data layout, gold lists, MedDRA folder expectations |
| [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md) | Full tree-oriented map of the repo |

---

## 9. MedDRA licensing

MedDRA is **not** redistributed in this repository. Obtain a licence and data from [meddra.org/software-packages](https://www.meddra.org/software-packages), then load into Neo4j per your internal graph build (`scripts/extract_meddra.py`, `build_graph.py` — see `data/README.md`).

---

## Full narrative

High-level conclusions (problem, research questions, stages, metrics): [`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md).

Methodology and evaluation plan (when present): [`docs/project-plan/TermPlanMT_Proposal.pdf`](docs/project-plan/TermPlanMT_Proposal.pdf).
