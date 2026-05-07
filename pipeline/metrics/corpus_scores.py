"""Corpus BLEU / chrF (used by ``tools/eval/evaluate.py`` and ``tools/eval/plot_figures.py``)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

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


# Join segment strings when forming one hypothesis / reference per document (BLEU doc-concat).
DOC_CONCAT_SEPARATOR = " "


def macro_bleu_doc_concat(
    hyps: list[str],
    refs: list[str],
    groups: list[str],
    *,
    sep: str = DOC_CONCAT_SEPARATOR,
) -> float:
    """Unweighted mean of corpus BLEU on **concatenated** segment text per document.

    For each document key in ``groups``, segment ``hyps`` / ``refs`` are taken in
    **input list order** (callers should sort by segment id before calling). Each
    document yields one synthetic sentence pair ``sep.join(hyps_i)`` vs
    ``sep.join(refs_i)``, scored with :func:`corpus_bleu` as a one-line corpus.
    """
    if not hyps or len(hyps) != len(refs) or len(hyps) != len(groups):
        return float("nan")
    idx_by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by[str(g)].append(i)
    scores: list[float] = []
    for indices in idx_by.values():
        doc_h = sep.join(str(hyps[j] or "") for j in indices)
        doc_r = sep.join(str(refs[j] or "") for j in indices)
        scores.append(corpus_bleu([doc_h], [doc_r]))
    return sum(scores) / len(scores) if scores else float("nan")


def macro_corpus_metric_by_group(
    corpus_fn: Callable[[list[str], list[str]], float],
    hyps: list[str],
    refs: list[str],
    groups: list[str],
) -> float:
    """Unweighted mean of ``corpus_fn`` applied separately to each document group.

    ``groups[i]`` is the document key aligned with ``hyps[i]`` / ``refs[i]``.
    """
    if not hyps or len(hyps) != len(refs) or len(hyps) != len(groups):
        return float("nan")
    idx_by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by[str(g)].append(i)
    scores: list[float] = []
    for indices in idx_by.values():
        hs = [hyps[j] for j in indices]
        rs = [refs[j] for j in indices]
        scores.append(corpus_fn(hs, rs))
    return sum(scores) / len(scores) if scores else float("nan")
