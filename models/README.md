# Model checkpoints (local only)

Large weights are **not** committed. After training or downloading, expect paths such as:

| Path | Role |
|------|------|
| `biomistral-ner-lora/` | LoRA adapter (Unsloth) |
| `biomistral-ner-merged/` | Merged BioMistral-7B for `--backend unsloth` NER inference |
| `camembert-quaero-ner/` | Optional CamemBERT token-classification baseline (separate experiment) |

## Fine-tuned NER used in the paper

See **[`docs/NER_FINETUNING.md`](../docs/NER_FINETUNING.md)** for:

- **1,540** QUAERO BRAT sentence-level training examples (EMEA + MEDLINE train)
- LoRA hyperparameters and evaluation segment file
- Downstream +7 BLEU effect on S3/S4/S5-Mistral

Training requires `data/QUAERO_FrenchMed/` (gitignored) and GPU memory suitable for BioMistral-7B QLoRA. Re-run NER on Section 4.8 segments and write `data/section48/segments_ner_unsloth_full.jsonl` before `./rerun_all.sh` for the finetuned results tree.
