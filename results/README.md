## Results directories (canonical meaning)

This repo uses **two** main result trees that share the same structure.

### `ner_biollm/` — LOW-CCR baseline

- **Meaning**: baseline condition with lower grounding coverage (CCR bottleneck).
- **Canonical segments file**: `data/section48/segments_ner_biollm.jsonl`
- **Figures / tables**: `results/ner_biollm/figures/`

### `ner_biollm_finetuned/` — fine-tuned (higher-quality) NER condition

- **Meaning**: condition using the fine-tuned NER spans (cleaner boundaries; fewer artefacts than baseline NER).
- **Canonical segments file**: `data/section48/segments_ner_unsloth_full.jsonl`
- **Figures / tables**: `results/ner_biollm_finetuned/figures/`

Notes:
- Large intermediate `*.jsonl` outputs may be gitignored; regenerate with `rerun_all.sh` and the scripts under `tools/`.

