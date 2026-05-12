# Summary metrics

| System | chrF++ | HTM (lex) | BLEU | doc-BLEU | doc-chrF | BLEU† | Mean s/seg | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 NLLB | 31.90 | 0.195 | 18.89 | 18.89 | 31.90 | 43.82 | 0.77 | 3.81 |
| S2 Mistral (doc) | 35.85 | 0.274 | 20.37 | 20.37 | 35.85 | 46.26 | 8.04 | 17.91 |
| S3 GraphRAG | 32.01 | 0.246 | 17.49 | 17.49 | 32.01 | 39.60 | 3.55 | 16.01 |
| S4 rerank | 32.04 | 0.246 | 17.52 | 17.52 | 32.04 | 39.81 | 11.21 | 42.32 |
| S5 NLLB + boost | 31.02 | 0.205 | 17.54 | 17.54 | 31.02 | 41.35 | 0.89 | 3.93 |
| S5 Mistral + boost | 32.35 | 0.248 | 17.50 | 17.50 | 32.35 | 39.49 | 3.54 | 16.05 |

*BLEU† — macro mean over documents of corpus BLEU on a single synthetic line per document (segment `hyp` / `en_ref` joined with spaces; column `bleu_doc_concat` in CSV).*

*rHTM (dataset, gold `en_ref` vs grounded MedDRA English): 0.180*
