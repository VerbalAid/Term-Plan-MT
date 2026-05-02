# Ambiguous French grounding (multi-concept MedDRA keys)

During **string** grounding, `TermGraph` loads all French `fr_label` values from Neo4j and groups them by a **normalised** key (`normalize_fr_for_grounding`). If that key maps to **more than one distinct concept id**, the key is marked ambiguous: the pipeline still picks one node (exact hit or fuzzy), but logs:

`Ambiguous FR grounding: '<surface>' maps to N distinct MedDRA concepts under normalised key.`

This is a **graph / lexicon** phenomenon (synonymous or colliding FR labels), not an NER-model bug by itself. It matters for error analysis because the chosen concept may swap between runs or conditions if ranking differs.

## What I compared across NER conditions

1. **Coverage:** For each condition’s segment JSONL, I counted NER spans whose normalised `word` falls on an ambiguous key. Higher counts mean the extractor feeds more mass into collision-prone vocabulary.
2. **Qualitative:** For each key, I inspected example French sentences (`fr_snippet_*` in the CSV) and the MedDRA PT/LLT set (Neo4j queries or SmPC context) to see *why* two concepts share the same normalised FR surface.
3. **Planning / downstream:** Stage 3 locks are keyed by the **exact** NER surface string (`word`), not the normalised key. The report marks `has_planning_lock_*` if **any** surface form for that key appears in `planning_locks.json`. Where locks are missing, `graph.ground` may still have returned a concept for S2/S3 prompts, but global consistency across segments is weaker.

## Script

[`scripts/report_ambiguous_grounding.py`](../../scripts/report_ambiguous_grounding.py) writes `error_analysis/ambiguous_grounding_report.csv` and prints aggregate counts for two JSONLs (defaults: `segments_ner_biollm.jsonl` vs `segments_ner_unsloth.jsonl`).

**Stale locks:** After changing segments without regenerating `planning_locks.json`, I deleted the locks file or re-ran the pipeline with recompute so lock columns matched the current NER file.
