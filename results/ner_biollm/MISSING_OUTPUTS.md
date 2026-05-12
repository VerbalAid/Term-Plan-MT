## Missing system output JSONLs (ner_biollm)

This repo contains the **summary figures/CSVs** for the `ner_biollm` condition under:

- `results/ner_biollm/figures/`

However, the **system output JSONL files** (`s1.jsonl`, `s2.jsonl`, `s3.jsonl`, `s4.jsonl`, `s5.jsonl`, `s5_mistral.jsonl`) are **not present** under `results/ner_biollm/` in this working tree.

### What is present

- `results/ner_biollm/figures/scores_summary.csv`
- `results/ner_biollm/figures/paper_summary_table.md`
- other derived CSVs under `results/ner_biollm/figures/`

### What is absent (expected but missing)

- `results/ner_biollm/s1.jsonl`
- `results/ner_biollm/s2.jsonl`
- `results/ner_biollm/s3.jsonl`
- `results/ner_biollm/s4.jsonl`
- `results/ner_biollm/s5.jsonl`
- `results/ner_biollm/s5_mistral.jsonl`

### Why this matters

Some qualitative analyses (e.g. drift tables that quote exact `hyp` strings per segment) normally require the per-system JSONLs. In the current repo state, those drift examples must be taken from the available condition:

- `results/ner_biollm_finetuned/` (which includes `s1.jsonl`…`s5_mistral.jsonl`)

### Regeneration notes (no-GPU constraint)

- `rerun_all.sh` can regenerate outputs, but **S3/S4/S5_mistral require Mistral inference**, which is typically a GPU workload.
- Under a strict “no GPU” environment, only CPU-feasible stages (e.g. NLLB-based systems) could be regenerated, and only if the environment has the required models available.

### Bootstrap ΔBLEU (Table 5)

`tools/eval/bootstrap_bleu_delta.py` needs the **same** `s*.jsonl` files listed above under `results/ner_biollm/`. If they are missing, the script prints `[skip]` and does not write `bleu_delta_bootstrap_95ci.csv`.

After regenerating (or restoring from backup) those JSONLs, use the same segment file and exclusion policy as the paper tables, for example:

```bash
PYTHONPATH=. python tools/eval/bootstrap_bleu_delta.py \
  --baseline-dir results/ner_biollm \
  --finetuned-dir results/ner_biollm_finetuned \
  --segments data/section48/segments_ner_unsloth_full.jsonl \
  --exclude-segment-ids "" \
  --baseline-eval-file-set standard \
  --finetuned-eval-file-set mistral_clean \
  --n-bootstrap 2000 \
  --out-csv results/ner_biollm_finetuned/figures/bleu_delta_bootstrap_95ci.csv
```

See `tools/README.md` (bootstrap subsection) for details.

