# Summary metrics

| System | chrF++ | HTM (lex) | BLEU | doc-BLEU | doc-chrF | BLEU† | COMET | Mean s/seg | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 NLLB | 37.14 | 0.321 | 22.89 | 22.89 | 37.14 | 53.28 | — | 0.75 | 3.74 |
| S2 Mistral (doc) | 34.85 | 0.299 | 16.63 | 16.63 | 34.85 | 40.19 | — | 10.74 | 21.13 |
| S3 GraphRAG | 34.36 | 0.397 | 10.39 | 10.39 | 34.36 | 24.19 | — | 4.96 | 14.53 |
| S4 rerank | 34.24 | 0.429 | 9.93 | 9.93 | 34.24 | 23.02 | — | 16.57 | 44.68 |
| S5 NLLB + boost | 35.91 | 0.353 | 21.59 | 21.59 | 35.91 | 51.79 | — | 1.26 | 3.67 |
| S5 Mistral + boost | 34.32 | 0.380 | 10.32 | 10.32 | 34.32 | 24.18 | — | 12.45 | 38.84 |
| S6 NLLB + glossary | 35.84 | 0.315 | 19.56 | 19.56 | 35.84 | 46.22 | — | 0.75 | 2.06 |

*BLEU† — macro mean over documents of corpus BLEU on a single synthetic line per document (segment `hyp` / `en_ref` joined with spaces; column `bleu_doc_concat` in CSV).*

*rHTM (dataset, gold `en_ref` vs grounded MedDRA English): 0.151*
