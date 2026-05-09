# Data directory

| Path | Role |
|------|------|
| `meddra/` | MedDRA MedAscii (English hierarchy; not committed by default — see root `.gitignore`) |
| `ontology_ner_full_hierarchical_mistral_{train,val,test}.jsonl` | Mistral-instruct ontology SFT (one `{"text":...}` per line) |
| `ontology_ner_full_hierarchical_alpaca.jsonl` | Canonical hierarchical Alpaca ontology SFT (default for `biomistral_ner_finetune_unsloth.py --ontology-only`; 90/10 train/val split in memory). |
| `ontology_ner_full_hierarchical_mistral_train.jsonl.bak` | Optional pre-patch backup |
| `section48/` | Segment JSONLs for S1–S5 / NER (`segments_ner*.jsonl`) |

Regenerate ontology JSONL from Neo4j:

```bash
PYTHONPATH=. python tools/data/export_full_ontology_ner_sft_jsonl.py --prompt-style alpaca --out data/ontology_ner_full_hierarchical_alpaca.jsonl
PYTHONPATH=. python tools/data/export_full_ontology_ner_sft_jsonl.py --prompt-style mistral --out data/full_mistral.jsonl
PYTHONPATH=. python tools/data/split_ontology_sft_jsonl.py --input data/full_mistral.jsonl --out-dir data
# optional: rename mistral split outputs if you use separate train/val/test for Mistral
```

Fix `soc`…`llt` on existing hierarchical JSONL without Neo4j:

```bash
PYTHONPATH=. python tools/data/patch_ontology_sft_hierarchy_jsonl.py --input data/ontology_ner_full_hierarchical_mistral_train.jsonl --in-place --backup .bak
```
