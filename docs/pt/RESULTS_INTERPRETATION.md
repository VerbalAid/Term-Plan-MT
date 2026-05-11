# Interpretação dos resultados (números de referência)

**Idioma:** [English](../RESULTS_INTERPRETATION.md) · Português · [Deutsch](../de/RESULTS_INTERPRETATION.md)

Use este ficheiro ao redigir texto para artigos ou relatórios: **os números devem coincidir** com os [`scores_summary.csv`](../../results/ner_biollm/figures/scores_summary.csv) registados no repositório, gerados por `tools/eval/evaluate.py`, e não com texto informal de visões gerais anteriores.

## Onde estão as pontuações de referência

| Condição | Caminho |
|----------|---------|
| **ner_biollm** | [`results/ner_biollm/figures/scores_summary.csv`](../../results/ner_biollm/figures/scores_summary.csv) |
| **ner_biollm_finetuned** | [`results/ner_biollm_finetuned/figures/scores_summary.csv`](../../results/ner_biollm_finetuned/figures/scores_summary.csv) |

Tabelas legíveis (mesma fonte): [`paper_summary_table.md`](../../results/ner_biollm/figures/paper_summary_table.md) na pasta `figures/` de cada condição.

Volte a avaliar depois de alterar segmentos, o grafo Neo4j ou o código das métricas:

```bash
docker compose up -d   # Neo4j para CCR / HTM / métricas de grafo
PYTHONPATH=. python tools/eval/evaluate.py …   # ver tools/README.md
```

## Instantâneo alinhado com os CSV **commitados** (não arredonde de forma diferente noutros sítios)

Coluna lexical **`htm`** em `scores_summary.csv` (grounding ancorado em NER; cobertura reduzida explica valores absolutos baixos).

### ner_biollm (`results/ner_biollm/figures/scores_summary.csv`)

| Etapa | chrF++ | HTM (lex) | `htm_en_ref_dataset` | `ccr_dataset` |
|-------|--------|-----------|------------------------|---------------|
| S2 | 35.58 | **0.196** | 0.130 | 0.354 |
| S3 | 32.65 | **0.253** | 0.130 | 0.354 |

Nestes dados **HTM S3 > HTM S2** (não há “domínio do S2” em HTM). O chrF++ continua a favorecer **S2** face a S3–S5.

### ner_biollm_finetuned (`results/ner_biollm_finetuned/figures/scores_summary.csv`)

| Etapa | chrF++ | HTM (lex) | `htm_en_ref_dataset` | `ccr_dataset` |
|-------|--------|-----------|------------------------|---------------|
| S2 | 35.85 | **0.250** | 0.158 | 0.324 |
| S3 | 34.60 | **0.257** | 0.158 | 0.324 |

De novo **S3 ligeiramente à frente do S2 em HTM**; o pico de chrF++ mantém-se em **S2**.

## Visão geral vs discrepância nos CSV

Se alguma visão geral ou apresentação citar HTM à volta de **0.45 / 0.43** para S2/S3, isso **não corresponde** às tabelas commitadas acima (HTM lexical ≈0.20–0.26). Trate esses valores destacados como **avaliação mais antiga**, **variante métrica diferente** ou **erro**, até os reproduzir com `evaluate.py` e arquivar a configuração da execução.

## Porque o HTM parece baixo (contexto)

- O HTM lexical atual usa grounding **ancorado em NER** (`terms[].word` → Neo4j → comparar hipótese). Grounding escasso — logo o sinal HTM é fraco.
- **`htm_en_ref_dataset`** nos CSVs (~0.13–0.16 aqui) é um sinal de tipo limite superior sobre a frequência com que o inglês de referência se alinha com formulações estilo MedDRA na mesma engenharia — não substitui listas gold clínicas.

## Próximos passos priorizados (lista de trabalho)

1. **Reexecutar a avaliação** com Neo4j ativo e documentar revisão git + caminhos dos segmentos; atualizar `scores_summary.csv` se algo mudou.
2. **Lista de termos gold** — construir `gold_terms.json` (p.ex. via `build_gold_terms_from_parallel_ner.py`) para medir HTM face aos conceitos pretendidos, não só à cobertura NER.
3. **Folha qualitativa** — preencher etiquetas reais em [`error_analysis/error_review_50.csv`](../../error_analysis/error_review_50.csv); priorizar linhas de alta deriva em [`error_analysis/ner_biollm_term_drift.csv`](../../error_analysis/ner_biollm_term_drift.csv).
4. **Grounding ambíguo** — resolver casos concretos (p.ex. `pneumopathie inflammatoire`) com contexto MedDRA + locks opcionais de termos gold.
5. **Painel cross-NER** — executar [`tools/eval/plot_cross_ner_dashboard.py`](../../tools/eval/plot_cross_ner_dashboard.py) (ou a fase de avaliação em [`rerun_all.sh`](../../rerun_all.sh)) com Neo4j; o diretório de saída costuma ser `results/cross_ner_comparison/` (criado quando necessário, nem sempre commitado).
6. **Ontologia LoRA** — apêndice / trabalho futuro salvo tempo em cluster.

---

Atualize este documento sempre que fizer commit de novos `scores_summary.csv`, para manter texto e tabelas coerentes.
