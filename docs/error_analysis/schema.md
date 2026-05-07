# Manual error analysis — annotation schema

Use this for qualitative review of FR→EN hypotheses against references and graph-backed behaviour.

## Unit of annotation

One row per **(segment_id, system_id, issue)** triple. If a segment has multiple independent issues, use multiple rows with the same `segment_id` and `system_id`.

When exporting via `tools/error_analysis/sample_errors_for_annotation.py`, default sampling is **one CSV row per `segment_id`** (worst chrF triple among all systems/conditions); add `--repeat-segments` for other layouts.

## Fields

| Column | Required | Description |
|--------|----------|-------------|
| `segment_id` | yes | Matches `id` in `segments_ner*.jsonl` (e.g. `48_001`). |
| `ner_condition` | yes | Which NER setting produced this row (e.g. `ner_biollm`, `ner_biollm_finetuned`); folder under `results/`. |
| `system_id` | yes | `s1` … `s5`, or `s5_mistral`. |
| `issue_category` | yes | High-level bucket (see closed tag set below). |
| `severity` | optional | `minor` \| `major` \| `critical` (subjective; define with partner). |
| `source_span_fr` | optional | Problematic or relevant French substring (quote from segment `fr`). |
| `hypothesis_span_en` | optional | Corresponding hypothesis substring (quote from system output). |
| `ref_span_en` | optional | Reference substring from `en_ref` if contrast matters. |
| `wrong_terminology` | optional | `0` / `1` — (1) Wrong medico-regulatory concept or preferred English vs the reference (substitution, not just vague). |
| `concept_flattening_too_vague` | optional | `0` / `1` — (2) Specificity loss: hypothesis is broader/generic vs reference; related meaning but under-specific. |
| `missing_terms` | optional | `0` / `1` — (3) Meaningful reference wording absent or only weakly implied in the hypothesis. |
| `unnatural_phrasing` | optional | `0` / `1` — (4) Awkward or non-native SmPC English; concepts may still align. |
| `notes` | yes | Short free text: quote spans; reference which definitions (1–4) support each flag. |
| `reviewer` | optional | Initials or handle. |
| `resolved` | optional | `0` / `1` when tracking adjudication. |

### Suggested `issue_category` values (closed set — adjust once per paper)

- `terminology_wrong` — MedDRA-related term wrong or inconsistent with graph intent.
- `fluency` — grammar, awkwardness, non-native phrasing without wrong drug/ADR concept.
- `omission` — content dropped vs reference.
- `addition` — unjustified extra content.
- `ner_propagation` — error plausibly tied to missing/wrong NER span feeding S3–S5.
- `other` — use sparingly; extend list if patterns repeat.

Definitions (1–4) live in `tools/error_analysis/sample_errors_for_annotation.py` as `ERROR_REVIEW_ONTOLOGY_GUIDE` and are prepended to Ollama/OpenAI annotation prompts.

## Pairwise workflow

1. Export rows from `results/<ner_condition>/s*.jsonl` (see `export_template.csv` columns).
2. Sort by `segment_id`, compare systems side-by-side with `en_ref`.
3. Log disagreements with partner on `notes` + `resolved`.
