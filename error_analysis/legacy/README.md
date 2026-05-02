# Legacy error-analysis CSVs

CSVs from older NER conditions (e.g. CamemBERT baseline / finetuned tags). Regenerate current-condition samples with:

```bash
PYTHONPATH=. python scripts/sample_errors_for_annotation.py \
  --out-csv error_analysis/error_review_50.csv --n 50 --annotate-backend ollama
```
