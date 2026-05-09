# Training scripts (GPU-heavy)

This directory holds **French medical NER inference** and **Unsloth fine-tuning** helpers. The **core pipeline** (`tools/pipeline/run_pipeline.py`, `tools/eval/evaluate.py`, Neo4j, `pipeline/`) does not import these modules; it only needs segment JSONLs under `data/section48/`.

## Layout

| Path | Role |
| ---- | ---- |
| [`ner/`](ner/) | Prompted NER, NER fine-tuning, QUAERO BRAT loader. |
| `slurm_ontology_sft.sh` | Cluster run: ontology-only SFT. |
| `slurm_quaero_ner_sft.sh` | Cluster run: QUAERO-only NER fine-tune. |

## Environment

- Use the **same** project venv at the repo root: `.venv` and `pip install -r requirements.txt` (Unsloth / TRL blocks are commented in `requirements.txt`; follow the install order there for NER fine-tuning).
- **GPU:** NER inference and training need a CUDA device. Before 4-bit loads, scripts may require **several GiB free** VRAM (see comments in `biomistral_prompt_ner.py` and `biomistral_ner_finetune_unsloth.py`). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **before** launching Python when fragmentation is an issue.

## Commands (from repository root)

Set `PYTHONPATH=.` so `pipeline.*` imports resolve.

**Prompted BioMistral NER (writes e.g. `data/section48/segments_ner_biollm.jsonl`):**

```bash
cd "/path/to/MT_Project_Terminology "
PYTHONPATH=. ./.venv/bin/python training_scripts/ner/biomistral_prompt_ner.py --help
```

**Unsloth NER fine-tune / ontology SFT:**

```bash
PYTHONPATH=. ./.venv/bin/python training_scripts/ner/biomistral_ner_finetune_unsloth.py --help
```

Ontology SFT data comes from **`tools/data/export_full_ontology_ner_sft_jsonl.py`**. When **`data/ontology_ner_full_hierarchical_alpaca.jsonl`** exists, **`--ontology-only`** can omit `--ontology-sft-jsonl` / `--ontology-train-jsonl` / `--ontology-val-jsonl` (trainer loads that file and splits 90/10). Use **`tools/data/split_ontology_sft_jsonl.py`** only if you want separate train/val/test files on disk.

**Minimal Qwen ontology run + resume from latest checkpoint:**

```bash
PYTHONPATH=. ./.venv/bin/python training_scripts/ner/biomistral_ner_finetune_unsloth.py \
  --fit-8gb --ontology-only --resume-from-checkpoint auto --fast
```

`--fast` caps training at **one epoch**, skips the **per-epoch multiset F1** eval pass, and uses fewer log lines. **DataLoader workers stay 0** (Unsloth/TRL collators are not picklable for multiprocessing). For a very short smoke test, add e.g. **`--max-steps 500`**.

**Full ontology LoRA run** (no `--max-steps` cap; trains for **`--num-train-epochs`**, default **2**):

```bash
PYTHONPATH=. ./.venv/bin/python training_scripts/ner/biomistral_ner_finetune_unsloth.py \
  --fit-8gb --ontology-only --full-ontology-finetune --resume-from-checkpoint auto
```

**`--full-ontology-finetune` ignores `--fast` and clears `--max-steps`** if you passed either by habit.

**Try a saved LoRA checkpoint + dataset CCR** (requires GPU + Neo4j; pass output JSONL, LoRA `checkpoint-*` dir, and HF base id matching `adapter_config.json`):

```bash
bash training_scripts/ner/run_ner_ccr_from_lora_checkpoint.sh \
  data/section48/segments_ner_my_run.jsonl models/<your-lora>/checkpoint-5000 \
  unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit
```

**On ~8 GB GPUs** use `--fit-8gb` on the trainer (switches to `Qwen/Qwen2.5-3B-Instruct` and caps context). Default adapter dirs are described in `--help`.

## Copying segment JSONLs back

NER scripts write under `data/section48/` when given relative output paths. After generation, run the main pipeline from the repo root, for example:

```bash
PYTHONPATH=. ./.venv/bin/python tools/pipeline/run_pipeline.py --segments data/section48/segments_ner_biollm.jsonl --results-dir results/ner_biollm
```

Paper-facing tables and figures are produced by `tools/eval/evaluate.py`, `tools/eval/plot_figures.py`, and `rerun_all.sh` (see root `README.md`).
