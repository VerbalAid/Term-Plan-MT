# Model checkpoints (local)

Trained and merged weights (e.g. `biomistral-ner-merged`, ontology LoRA trees) live here but are **not committed to git** (multi‑GB). After cloning, copy `models/` from a backup or run training under `extras/experiments/french_medical_ner/` (see root `README.md`).

Hugging Face public base models are downloaded at runtime by the training scripts.

**Shipped example:** `biomistral-ner-merged/` may be present as the default `--unsloth-merged-path` for `biomistral_prompt_ner.py`.
