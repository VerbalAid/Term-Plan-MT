# NER ablations — narrative for the paper

## Primary quantitative finding (confirmed)

Using **the same JSON-list extraction prompt and evaluation pipeline**, **Mistral-7B-Instruct-v0.2** yields **CCR ≈ 0.222** on this MedDRA-grounded corpus, while **BioMistral-7B** yields **CCR ≈ 0.354**. Instruction tuning alone does **not** close the gap on French clinical span extraction; **biomedical pre-training** dominates here.

This **does not replicate** the broader HiTZ internship framing (*instruction following suffices*) **for this task**: medical argumentation ≠ medication terminology tagging under strict ontology grounding. Treat **instruction-only Mistral** as a deliberate negative control for domain specialisation.

## Interpretation

- **CCR** reflects whether extracted FR mentions successfully ground into MedDRA (`graph.ground`).
- Lower recall/precision on spans from Mistral-Instruct propagates directly into lower CCR; the difference is driven by **NER quality**, not translation quality.

## Caveats (methodological, not flaws in the headline result)

1. **Vector grounding must use real embeddings.** If `extras/experiments/vector_grounding/build_graph_embeddings.py` has not been run and the Neo4j vector index `meddra_fr_embedding` is missing, vector modes **must not** be trusted (historically they could report **CCR = 1.0** misleadingly). `extras/experiments/french_medical_ner/compare_neo4j_grounding_ccr.py` now refuses vector modes unless that index exists (**Neo4j 5:** scans `SHOW INDEXES`; older deployments fall back to `CALL db.indexes()` when available).
2. **Partial MT outputs invalidate BLEU/chrF/HTM.** Example: an interrupted **S4** run (`Predicting 3/16`) yields meaningless corpus scores; **`evaluate.py`** skips systems whose JSONL row count is below the segment file.
3. **Mistral JSON parse errors** (truncated lists) are mitigated in **`extras/experiments/french_medical_ner/biomistral_prompt_ner.py`** with a second greedy generation pass and a **larger** `max_new_tokens` budget on retry; document `JSON parse retries` in logs.
4. **Ollama** returns **HTTP 404** if the server is down or the tag is not pulled; the script now **exits** with a one-line fix (`ollama serve`, `ollama pull <model>`).
5. **Unsloth** fine-tuning requires `trl` + `datasets` + the `unsloth[...]` install (see `requirements.txt` and import error in `extras/experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py`). QUAERO BRAT loading: `extras/experiments/french_medical_ner/quaero_brat_reader.py`.

## Suggested table row (substitute exact digits from run logs)

| Condition | Model | CCR (dataset) | Notes |
|-----------|--------|----------------|--------|
| Biomed-pretrained | BioMistral-7B | **0.354** | Main NER row |
| Instruction-only | Mistral-7B-Instruct-v0.2 | **0.222** | NER ablation; domain gap |

Use locked numbers from `evaluate.py --ccr-only` or `compare_neo4j_grounding_ccr.py` (string mode) for the final digit.
