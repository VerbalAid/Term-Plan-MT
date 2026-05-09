# Fine-tuning Mistral-7B-Instruct for MedDRA ontology JSON (SFT)

## What you are building

A version of Mistral-7B-Instruct that, given a French MedDRA label, outputs structured ontology JSON (hierarchical Alpaca / Mistral `text` field). This repo also keeps the **S1–S5 translation ladder** under `tools/pipeline/run_pipeline.py` and `pipeline/systems/`.

---

### Alternative: ontology-only SFT (no segment sentences)

Build examples **only from Neo4j** (`:Concept` with `fr_label`, English `name`, hierarchy). Use `tools/data/export_full_ontology_ner_sft_jsonl.py` (see Step 1 below), then optionally `tools/data/split_ontology_sft_jsonl.py`. Train with `training_scripts/ner/biomistral_ner_finetune_unsloth.py`.

## Data quality before you export ontology JSONL

1. **Reload Neo4j after upgrading loaders** — `tools/data/build_graph.py` now decodes MedDRA ``*.asc`` as UTF-8 / CP1252 / ISO-8859-1 instead of UTF-8 with replacement characters, so French ``fr_label`` values stay intact (no U+FFFD replacement glyphs in exports).
2. **Hierarchy columns** — `TermGraph.fetch_hierarchy_for_concept` keys SOC→LLT off both Neo4j ``tier`` and ``level`` when ``tier`` is missing, so hierarchical JSONL gets non-null ``soc`` / ``pt`` on LLT rows when the graph path is complete.
3. **Mistral prompt framing** — For Mistral-7B-Instruct, regenerate with  
   ``python tools/data/export_full_ontology_ner_sft_jsonl.py --prompt-style mistral …``  
   then split as usual. That wraps each example as ``<s>[INST] … [/INST] …</s>`` instead of bare Alpaca section headers. For maximum fidelity you can still use ``tokenizer.apply_chat_template`` in your trainer on ``messages``-style rows.
4. **Patch old JSONL without re-querying Neo4j** — ``tools/data/patch_ontology_sft_hierarchy_jsonl.py`` rewrites path fields from ``mdhier.asc`` using **mdhier bucket tier** (not JSON ``tier``), normalizes numeric ``id`` forms, and fixes swapped code/name columns when the name slot is numeric. Re-run on your ``.bak`` if you patched with an earlier script version.

---

## Step 1 — Generate training data (on your laptop, no GPU needed)

You need Neo4j with MedDRA loaded (`tools/data/build_graph.py`). Export is **graph-only** (no segment JSONL):

```bash
PYTHONPATH=. python tools/data/export_full_ontology_ner_sft_jsonl.py \
  --prompt-style mistral --out data/ontology_ner_full_hierarchical_mistral.jsonl
PYTHONPATH=. python tools/data/split_ontology_sft_jsonl.py \
  --input data/ontology_ner_full_hierarchical_mistral.jsonl --out-dir data
```

Use the split `ontology_ner_full_hierarchical_*` JSONL files from the previous block. Training is done in-repo with Unsloth (see root `README.md`).

---

## Train (Unsloth, in-repo)

```bash
PYTHONPATH=. python training_scripts/ner/biomistral_ner_finetune_unsloth.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --ontology-only --full-ontology-finetune
```

Defaults pick up `data/ontology_ner_full_hierarchical_alpaca.jsonl` when present (90/10 train/val split in memory). For SLURM, see `training_scripts/slurm_ontology_sft.sh`.

To mix **QUAERO NER** data, install the BRAT corpus and pass `--brat-dir` (omit `--ontology-only`).

---

## After training

```bash
# On cluster
scp -r models/mistral-7b-instruct-v0-2-ner-lora your_laptop:~/path/to/repo/models/

# On laptop — smoke inference
export PYTHONPATH=.
PYTHONPATH=. python training_scripts/ner/biomistral_prompt_ner.py \
  --backend unsloth --unsloth-lora-path models/mistral-ontology-lora \
  --output data/section48/segments_smoke.jsonl
```

For full **S1–S5** runs on segment JSONL, use `tools/pipeline/run_pipeline.py` and `tools/eval/evaluate.py` (see root `README.md`).

---

## What success looks like

- Stable training loss and sensible generations on held-out ontology rows
- JSON list in the response parses cleanly (valid keys / MedDRA ids)
- Optional: run `tools/eval/evaluate.py` on the same segments for BLEU/chrF/COMET/CCR-style metrics (needs Neo4j for graph metrics unless `--no-graph`).
