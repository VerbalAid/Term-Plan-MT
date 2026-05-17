# Data directory

## Segment JSONLs (paper inputs)

| File | Condition |
|------|-----------|
| `section48/segments_ner_biollm.jsonl` | BioMistral-7B prompted NER — baseline condition. |
| `section48/segments_ner_unsloth_full.jsonl` | Fine-tuned BioMistral NER — paper main condition. |
| `section48/gold_glossary.json` | FR→EN glossary for the S6 oracle ablation. |

Each segment JSONL has one JSON object per line with keys `id`, `fr`, `en_ref`, `terms`.

## QUAERO training corpus (fine-tuned NER)

The fine-tuned condition is trained on **QUAERO French Medical** BRAT data (not redistributed here):

- **Location (local):** `data/QUAERO_FrenchMed/` (gitignored)
- **Splits:** `corpus/train/EMEA` + `corpus/train/MEDLINE`
- **Labels:** `DISO`, `CHEM`, `PROC` only (not MedDRA hierarchy levels)
- **Size:** **1,540** sentence-level lines combined (706 EMEA + 834 MEDLINE when all train `.txt` lines are counted)

Re-count after download:

```bash
PYTHONPATH=. python tools/count_quaero_brat_sentences.py
```

See **[`docs/NER_FINETUNING.md`](../docs/NER_FINETUNING.md)** for hyperparameters, evaluation effects, and what *not* to claim on posters.

## MedDRA setup (Neo4j graph)

MedDRA is not redistributed. Obtain a licence from the MSSO, then:

```bash
# Extract ASCII files from the MedDRA zip
PYTHONPATH=. python data/extract_meddra.py --meddra-dir data/meddra

# Load into Neo4j (docker compose up -d first)
PYTHONPATH=. python data/build_graph.py
```

`data/meddra/` is gitignored. The graph must be running (Neo4j) for any script that uses `TermGraph` (pipeline S3–S6, evaluation HTM/CCR).

## S6 glossary

`section48/gold_glossary.json` is the FR→EN glossary for the S6 oracle ablation.

**Preferred — LLM extraction via Ollama** (better quality):

```bash
# Requires: ollama serve && ollama pull mistral
python data/build_glossary.py \
  --input data/section48/segments_ner_biollm.jsonl \
  --output data/section48/gold_glossary.json \
  --model mistral --verbose

# Dry-run to inspect the prompt first:
python data/build_glossary.py --input data/section48/segments_ner_biollm.jsonl --dry-run
```

Batch size of 4 is the sweet spot — small enough the model stays focused, large enough to finish 126 pairs in ~32 calls. Multiple English renderings of the same French term are kept deliberately (both are valid for logit boosting).

**Fallback — n-gram heuristic** (no Ollama needed):

```bash
PYTHONPATH=. python data/build_gold_terms_from_parallel_ner.py \
  --segments data/section48/segments_ner_biollm.jsonl \
  --out data/section48/gold_glossary.json
```
