# Data directory

## Segment JSONLs (paper inputs)

| File | Condition |
|------|-----------|
| `section48/segments_ner_biollm.jsonl` | BioMistral-7B prompted NER — baseline condition. |
| `section48/segments_ner_unsloth_full.jsonl` | Fine-tuned BioMistral NER — paper main condition. |
| `section48/gold_glossary.json` | FR→EN glossary for the S6 oracle ablation. |

Each segment JSONL has one JSON object per line with keys `id`, `fr`, `en_ref`, `terms`.

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

`section48/gold_glossary.json` is a draft FR→EN glossary for the S6 oracle ablation. To regenerate from the segment corpus:

```bash
PYTHONPATH=. python data/build_gold_terms_from_parallel_ner.py \
  --segments data/section48/segments_ner_biollm.jsonl \
  --out data/section48/gold_glossary.json
```
