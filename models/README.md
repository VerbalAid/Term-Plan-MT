# Model checkpoints (local)

Trained and merged weights (BioMistral NER, CamemBERT, etc.) live here but are **not committed to git** (multi‑GB). After cloning, either:

- copy an existing `models/` tree from a backup, or  
- run the training / merge scripts under `extras/experiments/french_medical_ner/` (see root `README.md` and `extras/README.md`).

Hugging Face **public** NER baselines (e.g. `Jean-Baptiste/camembert-ner`) are pulled at runtime and do not need to be stored in this folder.

**Local pruning (2026-05):** removed redundant LoRA dirs, older CamemBERT checkpoints, and **local Qwen weight trees** (not referenced by `rerun_all.sh` / `tools/pipeline/run_pipeline.py`; regenerate with `extras/.../biomistral_ner_finetune_unsloth.py` + merge if needed). **Kept for the default thesis NER path:** `biomistral-ner-merged` (default `--unsloth-merged-path` in `biomistral_prompt_ner.py`) plus CamemBERT finetune dirs with latest checkpoints only.
