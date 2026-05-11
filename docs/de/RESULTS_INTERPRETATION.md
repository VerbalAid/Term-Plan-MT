# Ergebnisinterpretation (maßgebliche Zahlen)

**Sprache:** [English](../RESULTS_INTERPRETATION.md) · [Português](../pt/RESULTS_INTERPRETATION.md) · Deutsch

Nutzen Sie diese Seite für Texte zu Papers oder Berichten: **Die Zahlen müssen** mit den eingecheckten [`scores_summary.csv`](../../results/ner_biollm/figures/scores_summary.csv)-Dateien übereinstimmen, die von `tools/eval/evaluate.py` erzeugt wurden — nicht mit älterem Überblicksprosa.

## Wo die maßgeblichen Scores liegen

| Bedingung | Pfad |
|-----------|------|
| **ner_biollm** | [`results/ner_biollm/figures/scores_summary.csv`](../../results/ner_biollm/figures/scores_summary.csv) |
| **ner_biollm_finetuned** | [`results/ner_biollm_finetuned/figures/scores_summary.csv`](../../results/ner_biollm_finetuned/figures/scores_summary.csv) |

Lesbare Tabellen (gleiche Quelle): [`paper_summary_table.md`](../../results/ner_biollm/figures/paper_summary_table.md) im jeweiligen Ordner `figures/`.

Nach Änderungen an Segmenten, Neo4j-Graphen oder Metrikcode neu auswerten:

```bash
docker compose up -d   # Neo4j für CCR / HTM / Graph-Metriken
PYTHONPATH=. python tools/eval/evaluate.py …   # siehe tools/README.md
```

## Momentaufnahme passend zu den **eingecheckten** CSVs (anderswo nicht anders runden)

Die lexikalische **`htm`**-Spalte in `scores_summary.csv` (NER-verankertes Grounding; geringe Abdeckung erklärt niedrige absolute Werte).

### ner_biollm (`results/ner_biollm/figures/scores_summary.csv`)

| Stufe | chrF++ | HTM (lex) | `htm_en_ref_dataset` | `ccr_dataset` |
|-------|--------|-----------|------------------------|---------------|
| S2 | 35.58 | **0.196** | 0.130 | 0.354 |
| S3 | 32.65 | **0.253** | 0.130 | 0.354 |

Damit gilt **S3-HTM > S2-HTM** in diesem Stand (kein „S2 dominiert“ bei HTM). chrF++ begünstigt weiterhin **S2** gegenüber S3–S5.

### ner_biollm_finetuned (`results/ner_biollm_finetuned/figures/scores_summary.csv`)

| Stufe | chrF++ | HTM (lex) | `htm_en_ref_dataset` | `ccr_dataset` |
|-------|--------|-----------|------------------------|---------------|
| S2 | 35.85 | **0.250** | 0.158 | 0.324 |
| S3 | 34.60 | **0.257** | 0.158 | 0.324 |

Wiederum liegt **S3 bei HTM knapp vor S2**; das chrF++-Maximum bleibt bei **S2**.

## Überblick vs. CSV-Widerspruch

Wenn irgendein Überblick oder Folien HTM um **0,45 / 0,43** für S2/S3 nennt, **stimmt das nicht** mit den obigen eingecheckten Tabellen überein (lexikales HTM ≈0,20–0,26). Behandeln Sie solche Schlagzahlen als **ältere Auswertung**, **andere Metrikvariante** oder **Fehler**, bis Sie sie mit `evaluate.py` reproduziert und die Laufkonfiguration archiviert haben.

## Warum HTM niedrig wirkt (Kontext)

- Aktuelles lexikalisches HTM nutzt **NER-verankertes** Grounding (`terms[].word` → Neo4j → Hypothese vergleichen). Spärliches Grounding ⇒ schwaches HTM-Signal.
- **`htm_en_ref_dataset`** in den CSVs (~0,13–0,16 hier) ist ein Signal ähnlich einer Obergrenze dafür, wie oft die englische Referenz unter derselben Mechanik mit MedDRA-nahen Formulierungen übereinstimmt — kein Ersatz für klinische Goldlisten.

## Priorisierte nächste Schritte (Arbeitscheckliste)

1. **Auswertung erneut** mit laufendem Neo4j und dokumentierter Git-Revision + Segmentpfade; `scores_summary.csv` aktualisieren, falls sich etwas ändert.
2. **Gold-Termliste** — `gold_terms.json` aufbauen (z. B. über `build_gold_terms_from_parallel_ner.py`), damit HTM gegen beabsichtigte Konzepte und nicht nur NER-Abdeckung gemessen wird.
3. **Qualitatives Blatt** — echte Labels in [`error_analysis/error_review_50.csv`](../../error_analysis/error_review_50.csv); Zeilen mit hoher Drift aus [`error_analysis/ner_biollm_term_drift.csv`](../../error_analysis/ner_biollm_term_drift.csv) priorisieren.
4. **Mehrdeutiges Grounding** — konkrete Fälle klären (z. B. `pneumopathie inflammatoire`) mit MedDRA-Kontext + optional Gold-Term-Sperren.
5. **Cross-NER-Dashboard** — [`tools/eval/plot_cross_ner_dashboard.py`](../../tools/eval/plot_cross_ner_dashboard.py) (oder Eval-Phase in [`rerun_all.sh`](../../rerun_all.sh)) mit Neo4j ausführen; Ausgabe meist `results/cross_ner_comparison/` (bei Bedarf erzeugt, nicht immer eingecheckt).
6. **Ontologie-LoRA-Spur** — Anhang / weiterführende Arbeit, außer Clusterzeit ist frei.

---

Dieses Dokument bei neuen Commits von `scores_summary.csv` aktualisieren, damit Text und Tabellen zusammenpassen.
