# Model checkpoints (local)

Trained and merged weights (BioMistral NER, CamemBERT, etc.) live here but are **not committed to git** (multi‑GB). After cloning, either:

- copy an existing `models/` tree from a backup, or  
- run the training / merge scripts under `experiments/french_medical_ner/` (see root `README.md` and `experiments/README.md`).

Hugging Face **public** NER baselines (e.g. `Jean-Baptiste/camembert-ner`) are pulled at runtime and do not need to be stored in this folder.
