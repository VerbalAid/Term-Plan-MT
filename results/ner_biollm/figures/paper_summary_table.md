# Summary metrics

| System | chrF++ | HTM (lex) | BLEU | doc-BLEU | doc-chrF | BLEU† | Mean s/seg | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 NLLB | 31.90 | 0.114 | 18.89 | 18.89 | 31.90 | 43.82 | 0.78 | 3.81 |
| S2 Mistral (doc) | 35.58 | 0.196 | 19.84 | 19.84 | 35.58 | 45.55 | 8.60 | 23.97 |
| S3 GraphRAG | 32.65 | 0.253 | 10.79 | 10.79 | 32.65 | 26.06 | 5.29 | 16.71 |
| S4 rerank | 32.85 | 0.237 | 10.87 | 10.87 | 32.85 | 26.24 | 15.64 | 47.17 |
| S5 NLLB + boost | 30.55 | 0.135 | 17.34 | 17.34 | 30.55 | 41.51 | 0.91 | 4.17 |
| S5 Mistral + boost | 32.64 | 0.267 | 10.82 | 10.82 | 32.64 | 26.09 | 5.40 | 16.57 |

*BLEU† — macro mean over documents of corpus BLEU on a single synthetic line per document (segment `hyp` / `en_ref` joined with spaces; column `bleu_doc_concat` in CSV).*

*rHTM (dataset, gold `en_ref` vs grounded MedDRA English): 0.130*
