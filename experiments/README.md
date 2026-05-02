# Supplementary experiments

Supplementary code only (French medical NER, vector grounding, training curves). Paper-facing tables use **`results/ner_biollm/`** and **`results/ner_biollm_finetuned/`** only (see root `rerun_all.sh`). Engineering notes: `docs/PROJECT_NOTES.md`.

| Area | Location |
| ---- | -------- |
| French medical NER, Unsloth fine-tuning, QUAERO BRAT I/O, Neo4j grounding / CCR ablations | `experiments/french_medical_ner/` |
| Vector embeddings / vector CCR reports | `experiments/vector_grounding/` |
| Training curve overlays | `experiments/figures/` |
| Non–main-paper experiment outputs | Created locally under `experiments/*/results/` when running supplementary pipelines (not shipped in git). |

Paper-facing outputs use **`results/ner_biollm/`** (BioMistral prompt NER) and **`results/ner_biollm_finetuned/`** (when Unsloth segments exist). **`rerun_all.sh`** runs only these two. Cross-NER figures: **`results/cross_ner_comparison/`**.
