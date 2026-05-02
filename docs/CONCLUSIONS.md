# Conclusions — TermPlan-MT (graph-grounded terminology planning)

This note summarises what **TermPlan-MT** set out to do, how it maps onto **this repository**, and where the **implementation** matches or refines the [project proposal PDF](project-plan/TermPlanMT_Proposal.pdf) (when present under `docs/project-plan/`). It is written for readers who want the **thesis-level story** without re-reading the full proposal.

---

## 1. Problem statement (why TermPlan-MT exists)

Neural machine translation is often **fluent but under-constrained** for **regulatory medical text**. In SmPC-style documents the same source concept should surface **consistently** in English across many segments; inconsistency can carry **patient-safety** risk. The proposal highlights three recurring failure modes:

| Failure mode | Meaning (informal) |
| ------------ | ------------------- |
| **Term drift** | The same French cue is translated with **different** English wordings across the document. |
| **Concept flattening** | A **specific** MedDRA-level rendering is replaced by a **generic** hypernym (loss of clinical specificity). |
| **Coverage gaps** | Domain terms are **missed**, **under-extracted**, or **never grounded**, so downstream stages cannot constrain them. |

**MedDRA** supplies a validated **five-level** hierarchy (SOC → … → LLT). **TermPlan-MT** connects **French SmPC text**, **NER-derived spans**, and that **ontology** so that terminology decisions are **traceable** (Neo4j nodes, levels, neighbours) rather than opaque hidden states in a single large model.

---

## 2. Scope (what this corpus is)

- **Language pair:** French → English.  
- **Document:** EMA **KEYTRUDA (pembrolizumab)** SmPC material; evaluation focuses on **Section 4.8** (*Undesirable effects*) — the densest adverse-event terminology in the document.  
- **In this repo:** Section 4.8 is segmented into **127 aligned sentence pairs** for the default PDFs (see [`scripts/prepare_data.py`](../scripts/prepare_data.py) and [`data/README.md`](../data/README.md)). Default `rerun_all.sh` evaluation usually scores **126** of those (segment **`48_028`** excluded as the dense table block).  
- **MedDRA access:** the proposal assumes a **non-commercial academic licence** via MSSO; the graph is **not** shipped here — see the root [`README.md`](../README.md) §9 and [`data/README.md`](../data/README.md).

---

## 3. System architecture (four stages → code)

The proposal’s **Stage 1–4** pipeline matches the mental model in the root [`README.md`](../README.md):

| Proposal stage | Role | Primary code / artefacts |
| -------------- | ---- | ------------------------ |
| **1 — NER** | Extract French medical **spans** (`terms[]` in JSONL). | [`experiments/french_medical_ner/biomistral_prompt_ner.py`](../experiments/french_medical_ner/biomistral_prompt_ner.py) (prompted BioMistral), [`experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py`](../experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py) (fine-tuned NER); historically CamemBERT/QUAERO paths are **archived** (see [`appendix_historical_ner_and_pipeline_results.md`](appendix_historical_ner_and_pipeline_results.md)). |
| **2 — Grounding** | Map each span to a **MedDRA concept** in Neo4j. | [`pipeline/graph.py`](../pipeline/graph.py) (`TermGraph.ground`, string / vector / vector+LLM modes). **CCR** = grounded ÷ extracted spans ([`pipeline/metrics/ccr.py`](../pipeline/metrics/ccr.py)). |
| **3 — Planning** | Choose **canonical English renderings** and **global locks** before decoding. | [`pipeline/planning.py`](../pipeline/planning.py); optional materialised [`planning_locks.json`](../data/section48/planning_locks.json) beside segment files. |
| **4 — MT** | **Five-condition** ladder **S1–S5**: NLLB (S1), long-context Mistral (S2), GraphRAG (S3), graph-informed rerank (S4), logit-boost decoding (S5); S5 can run **NLLB and/or Mistral** decoders → `s5.jsonl` / `s5_mistral.jsonl`. | [`pipeline/systems/`](../pipeline/systems/) (`s1_nllb` … `s5_logit`), driven by [`scripts/run_pipeline.py`](../scripts/run_pipeline.py); outputs under `results/<condition>/`. |

The **Neo4j graph** is the **interpretable layer**: each commitment ties to a concept node, its **tier/level**, and graph **structure**, not only a flat glossary lookup.

---

## 4. Research questions (proposal → measured answers)

**Ladder:** S1…S5 isolate context, graph-in-prompt retrieval, reranking, and logit-boost decoding ([`run_pipeline.py`](../scripts/run_pipeline.py)). **Metrics:** fluency (BLEU, chrF++, optional COMET), **HTM** (gold French substring in `fr`; [`htm.py`](../pipeline/metrics/htm.py)), **CCR** ([`ccr.py`](../pipeline/metrics/ccr.py)). **TCR** is not implemented; [`scripts/evaluate.py`](../scripts/evaluate.py) reports fluency, HTM, and CCR only.

**Numbers** (prompt BioMistral NER, string Neo4j grounding, **`48_028` excluded**): [`results/ner_biollm/figures/scores_summary.csv`](../results/ner_biollm/figures/scores_summary.csv) (HTM column from an evaluation-time gold list; re-scoring needs `--gold-terms`); see [`interpretation_of_results_snapshots.md`](interpretation_of_results_snapshots.md) and [`RESULTS_INTERPRETATION.md`](RESULTS_INTERPRETATION.md).

| # | Question | Contrast | Measured pattern (snapshot) |
| - | -------- | -------- | ---------------------------- |
| 1 | Does graph-backed planning beat bare NMT on terminology? | S3–S5 vs **S1** | **HTM** rises on graph-heavy Mistral lines (**S3/S4/S5 Mistral 0.435** vs **S1 0.370**) at a **large BLEU/chrF cost** on S3–S4 (e.g. S1 BLEU **22.9** vs S3 **10.8**). Uptake still **CCR-bounded** (~**0.35** dataset CCR on this slice). |
| 2 | Does long-context LLM help without ontology stack? | **S2** vs **S1** | **chrF** and **HTM** up (**39.7** / **0.457** vs **37.1** / **0.370**), **BLEU** down (**18.8** vs **22.9**) — better reference overlap and hierarchy score on gold hits, not a uniform “fluency win.” |
| 3 | Does GraphRAG (structure, no decode hardening) move terminology? | **S3** vs **S2** | On this run **S2 stays ahead** on chrF and HTM (**0.457** vs **0.435**). Graph-in-prompt **lifts HTM vs S1** but **does not beat doc-context Mistral** here; fluency drops sharply vs S1/S2. |
| 4 | Does reranking beat plain GraphRAG? | **S4** vs **S3** | **Same HTM** (**0.435**); tiny BLEU/chrF movement. Matches narrative: **small marginal effect** vs latency cost. |
| 5 | Does logit-boost decoding beat reranking alone? | **S5** vs **S4** | **S5 Mistral** matches **S4** on HTM (**0.435**); **S5 NLLB** HTM back to **S1** level (**0.370**) — **no extra hierarchy gain** over S3/S4 on string HTM for the Mistral branch in this table. |

**S3** is still the cleanest “**graph signal without decode forcing**” stage in design terms; **measurement** says gains over S1 are real on HTM, but **S2 already sets a strong HTM/chrF bar**, and **S4/S5** mostly **plateau or trade time**, not leapfrog S3 on HTM in this snapshot.

---

## 5. Metrics (what is measured)

| Concern | Metric | Tooling in repo |
| ------- | ------ | ---------------- |
| Fluency vs reference | **BLEU**, **chrF++**, optional **COMET** | [`scripts/evaluate.py`](../scripts/evaluate.py), [`scripts/plot_results.py`](../scripts/plot_results.py) |
| Hierarchy / gold-triggered terminology | **HTM** (1.0 / 0.5 / 0.0 per scored gold hit when the gold French cue appears in `fr`) | [`pipeline/metrics/htm.py`](../pipeline/metrics/htm.py); pass **`--gold-terms`** FR→EN JSON to [`evaluate.py`](../scripts/evaluate.py) / [`plot_results.py`](../scripts/plot_results.py) |
| Extracted span → graph reachability | **CCR** (dataset) | [`pipeline/metrics/ccr.py`](../pipeline/metrics/ccr.py) |

**HTM** is the proposal’s **hierarchy-aware** score: for each **applicable** gold row (French cue present in the segment source), it asks whether English in the hypothesis **respects** the intended MedDRA **level/branch** relative to that row and Neo4j (`same_branch`, level match), not merely whether a synonym string appears somewhere in the hypothesis. It does **not** measure “specificity drift” from French to English without that gold list and graph.

---

## 6. Gold annotation (evaluation contract)

The proposal describes a **manual FR→EN term mapping** with **MedDRA levels** checked against the official browser. **HTM** in this codebase expects that same style of **reviewed** list as a JSON file you pass with **`--gold-terms`** (not shipped in-repo).

---

## 7. Contributions (what TermPlan-MT adds)

- **Interpretable terminology layer:** French spans → **Neo4j** concepts → **locks** → MT — each step inspectable.  
- **Causal ablation ladder:** S1→S5 isolates **context**, **graph-informed prompting**, **reranking**, and **decoding-time** pressure.  
- **HTM:** explicitly scores **hierarchy drift** against gold + graph structure (see root README Figure 2).  
- **Reusable graph workflow:** same Neo4j store can accumulate **multiple** SmPCs under your licence and ingest pipeline.  
- **NER flexibility:** prompted vs fine-tuned **BioMistral** extractors ([`experiments/french_medical_ner/`](../experiments/french_medical_ner/)), with **grounding-mode** studies ([`compare_neo4j_grounding_ccr.py`](../experiments/french_medical_ner/compare_neo4j_grounding_ccr.py)) supporting methodology appendices.

---

## 8. Limitations (proposal + engineering reality)

- **MedDRA coverage:** terms **outside** MedDRA cannot be grounded; **CCR** quantifies how often extraction even reaches the graph.  
- **NER recall:** missed French spans **never** enter grounding or planning — pipeline quality is **bounded by Stage 1**.  
- **Fluency trade-off:** harder terminology enforcement can **reduce** BLEU/chrF-style overlap with a single reference; that is an **expected** tension, not necessarily an error.  
- **Scope:** one **Section 4.8** slice, one **language pair**, one **drug family** in the main path; generalisation is **future work**.  
- **Single reference:** BLEU/chrF can penalise **legitimate** regulatory paraphrase; COMET and **manual error analysis** complement automated scores.

---

## 9. Where to read next

| Document | Use |
| -------- | --- |
| [`README.md`](../README.md) | End-to-end pipeline, figures, navigation |
| [`RESULTS_INTERPRETATION.md`](RESULTS_INTERPRETATION.md) | Cross-NER narrative and figures |
| [`interpretation_of_results_snapshots.md`](interpretation_of_results_snapshots.md) | Prompt vs fine-tuned NER comparison tables |
| [`appendix_historical_ner_and_pipeline_results.md`](appendix_historical_ner_and_pipeline_results.md) | Historical NER conditions and archived metrics |
| [`paper_narrative_ner.md`](paper_narrative_ner.md) | NER-focused ablation narrative for writing |
| [`project-plan/TermPlanMT_Proposal.pdf`](project-plan/TermPlanMT_Proposal.pdf) | Full formal proposal (if committed) |
