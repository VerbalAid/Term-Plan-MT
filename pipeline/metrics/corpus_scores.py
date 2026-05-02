"""Corpus BLEU / chrF (shared by evaluate.py and plot_results.py)."""

from __future__ import annotations

try:
    from sacrebleu.metrics import BLEU, CHRF
except ImportError:
    from sacrebleu import corpus_bleu as _corpus_bleu, corpus_chrf as _corpus_chrf

    def corpus_bleu(hyps: list[str], refs: list[str]) -> float:
        return _corpus_bleu(hyps, [refs]).score

    def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
        return _corpus_chrf(hyps, [refs]).score

else:

    def corpus_bleu(hyps: list[str], refs: list[str]) -> float:
        return BLEU().corpus_score(hyps, [refs]).score

    def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
        return CHRF().corpus_score(hyps, [refs]).score
