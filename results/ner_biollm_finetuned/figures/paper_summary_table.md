# Summary metrics

| System | chrF++ | HTM (lex) | BLEU | doc-BLEU | doc-chrF | BLEU† | COMET | Mean s/seg | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 NLLB | 37.14 | 0.550 | 22.89 | 22.89 | 37.14 | 53.28 | — | 0.69 | 1.49 |
| S2 Mistral (doc) | 39.95 | 0.735 | 22.51 | 22.51 | 39.95 | 51.67 | — | 7.91 | 11.63 |
| S3 GraphRAG | 35.33 | 0.664 | 19.58 | 19.58 | 35.33 | 44.69 | — | 3.41 | 15.35 |
| S4 rerank | 35.36 | 0.664 | 19.66 | 19.66 | 35.36 | 45.06 | — | 10.81 | 39.63 |
| S5 NLLB + boost | 36.06 | 0.578 | 21.37 | 21.37 | 36.06 | 50.60 | — | 0.82 | 2.27 |
| S5 Mistral + boost | 35.78 | 0.673 | 20.61 | 20.61 | 35.78 | 46.86 | — | 3.40 | 15.69 |
| S6 NLLB + glossary | 35.48 | 0.517 | 19.25 | 19.25 | 35.48 | 46.07 | — | 0.82 | 2.43 |

*BLEU† — macro mean over documents of corpus BLEU on a single synthetic line per document (segment `hyp` / `en_ref` joined with spaces; column `bleu_doc_concat` in CSV).*

*rHTM (dataset, gold `en_ref` vs grounded MedDRA English): 0.178*
