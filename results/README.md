## Results directories (canonical meaning)

This repo uses **two** main result trees that share the same structure.

### `ner_biollm/` — LOW-CCR baseline

- **Meaning**: baseline condition with lower grounding coverage (CCR bottleneck).
- **Canonical segments file**: `data/section48/segments_ner_biollm.jsonl`
- **Figures / tables**: `results/ner_biollm/figures/`

### `ner_biollm_finetuned/` — fine-tuned NER condition

- **Meaning**: BioMistral-7B after Unsloth QLoRA on **~1,540** QUAERO BRAT sentences (see [`docs/NER_FINETUNING.md`](../docs/NER_FINETUNING.md)).
- **Canonical segments file**: `data/section48/segments_ner_unsloth_full.jsonl`
- **Figures / tables**: `results/ner_biollm_finetuned/figures/`
- **Key downstream effect**: S3/S4/S5-Mistral gain **~+7 BLEU** vs baseline NER (`bleu_delta_bootstrap_95ci.csv`); S1/S2 unchanged.

Notes:
- Large intermediate `*.jsonl` outputs may be gitignored; regenerate with `rerun_all.sh` and the root-level scripts (run_pipeline.py, evaluate.py, plot_figures.py).

