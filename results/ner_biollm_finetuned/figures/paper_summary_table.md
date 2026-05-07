# Summary metrics

| System | chrF++ | HTM (lex) | BLEU | doc-BLEU | doc-chrF | BLEU† | Mean s/seg | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 NLLB | 31.90 | 0.165 | 18.89 | 18.89 | 31.90 | 43.82 | 0.77 | 3.81 |
| S2 Mistral (doc) | 35.85 | 0.250 | 20.37 | 20.37 | 35.85 | 46.26 | 8.04 | 17.91 |
| S3 GraphRAG | 34.60 | 0.257 | 15.36 | 15.36 | 34.60 | 36.40 | 3.55 | 16.01 |
| S4 rerank | 34.92 | 0.257 | 15.66 | 15.66 | 34.92 | 37.19 | 11.21 | 42.32 |
| S5 NLLB + boost | 31.02 | 0.174 | 17.54 | 17.54 | 31.02 | 41.35 | 0.89 | 3.93 |
| S5 Mistral + boost | 34.85 | 0.261 | 15.98 | 15.98 | 34.85 | 37.82 | 3.54 | 16.05 |

*BLEU† — macro mean over documents of corpus BLEU on a single synthetic line per document (segment `hyp` / `en_ref` joined with spaces; column `bleu_doc_concat` in CSV).*

*rHTM (dataset, gold `en_ref` vs grounded MedDRA English): 0.158*
