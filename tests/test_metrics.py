"""Unit tests for HTM and matching helpers."""

from __future__ import annotations

import math

import pytest

from pipeline.metrics.htm import (
    compute_htm,
    compute_htm_en_ref,
    compute_htm_hyp_vs_ref,
    htm_vector_column_key,
    parse_cosine_thresholds_csv,
)
from pipeline.metrics.matching import normalize_text, phrase_in_hyp, phrase_in_text


def test_htm_vector_column_key_rounding() -> None:
    assert htm_vector_column_key(0.8) == "htm_vector_080"
    assert htm_vector_column_key(0.9) == "htm_vector_090"


def test_parse_cosine_thresholds_csv() -> None:
    assert parse_cosine_thresholds_csv("") == []
    assert parse_cosine_thresholds_csv("0.8, 0.9") == [0.8, 0.9]


def test_htm_miss_scores_zero() -> None:
    class _StubGraph:
        def ground(self, word: str, context=None):
            if word == "testspan":
                return {"name": "ExpectedEN", "level": 1}
            return None

        def candidate_renderings(self, _concept):
            return []

        def get_by_name(self, _name: str):
            return {"level": 1}

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    sid = "t1"
    id_to_segment = {
        sid: {
            "id": sid,
            "fr": "text with testspan inside",
            "terms": [{"word": "testspan"}],
        },
    }
    results = [{"id": sid, "hyp": "no matching english phrase here"}]
    assert compute_htm(results, _StubGraph(), id_to_segment) == 0.0


def test_normalize_hyphen_variants() -> None:
    a = normalize_text("immune-mediated")
    b = normalize_text("immune–mediated")
    c = normalize_text("immune mediated")
    assert a == b == c


def test_phrase_in_hyp_case_insensitive() -> None:
    assert phrase_in_hyp("Patient had PNEUMONIA", "pneumonia") is True
    assert phrase_in_hyp("patient had pneumonia", "PNEUMONIA") is True
    assert phrase_in_text("Same as hyp helper", "same as") is True


def test_htm_en_ref_matches_reference_surface() -> None:
    class _StubGraph:
        def ground(self, word: str, context=None):
            if word == "testspan":
                return {"name": "GoldPhrase", "level": 1}
            return None

        def candidate_renderings(self, _concept):
            return []

        def get_by_name(self, _name: str):
            return {"level": 1}

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    segment_rows = [
        {
            "id": "s1",
            "fr": "context testspan",
            "en_ref": "The GoldPhrase appears in the reference.",
            "terms": [{"word": "testspan"}],
        }
    ]
    assert compute_htm_en_ref(segment_rows, _StubGraph()) == 1.0


def test_compute_htm_hyp_vs_ref_perfect_agreement() -> None:
    class _StubGraph:
        def ground(self, word: str, context=None):
            if word == "spanfr":
                return {"name": "GoldPhrase", "level": 1}
            return None

        def candidate_renderings(self, _concept):
            return []

        def get_by_name(self, _name: str):
            return {"level": 1}

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    sid = "s1"
    id_to_segment = {
        sid: {
            "id": sid,
            "fr": "x spanfr y",
            "en_ref": "See GoldPhrase in reference.",
            "terms": [{"word": "spanfr"}],
        },
    }
    results = [{"id": sid, "hyp": "Also GoldPhrase in hypothesis."}]
    assert compute_htm_hyp_vs_ref(results, _StubGraph(), id_to_segment) == 1.0


def test_compute_htm_hyp_vs_ref_max_deviation() -> None:
    class _StubGraph:
        def ground(self, word: str, context=None):
            if word == "spanfr":
                return {"name": "GoldPhrase", "level": 1}
            return None

        def candidate_renderings(self, _concept):
            return []

        def get_by_name(self, _name: str):
            return {"level": 1}

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    sid = "s1"
    id_to_segment = {
        sid: {
            "id": sid,
            "fr": "x spanfr y",
            "en_ref": "The GoldPhrase appears.",
            "terms": [{"word": "spanfr"}],
        },
    }
    results = [{"id": sid, "hyp": "no gold phrase here"}]
    assert compute_htm_hyp_vs_ref(results, _StubGraph(), id_to_segment) == 0.0


def test_compute_htm_hyp_vs_ref_intermediate_agreement() -> None:
    """Reference matches a rendering with branch score 0.5; hyp misses -> agreement 0.5."""

    class _StubGraph:
        def ground(self, word: str, context=None):
            if word == "spanfr":
                return {"name": "MedPT", "level": 1}
            return None

        def candidate_renderings(self, _concept):
            return ["GoldPhrase"]

        def get_by_name(self, name: str):
            if name == "MedPT":
                return None
            if name == "GoldPhrase":
                return {"level": 99}
            return None

        def same_branch(self, _a: str, _b: str) -> bool:
            return True

    sid = "s1"
    id_to_segment = {
        sid: {
            "id": sid,
            "fr": "spanfr",
            "en_ref": "Regulatory GoldPhrase wording.",
            "terms": [{"word": "spanfr"}],
        },
    }
    results = [{"id": sid, "hyp": "No matching phrase."}]
    assert compute_htm_hyp_vs_ref(results, _StubGraph(), id_to_segment) == pytest.approx(0.5)


def test_compute_htm_hyp_vs_ref_nan_when_no_grounded_spans() -> None:
    class _StubGraph:
        def ground(self, word: str, context=None):
            return None

        def candidate_renderings(self, _concept):
            return []

        def get_by_name(self, _name: str):
            return None

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    sid = "s1"
    id_to_segment = {
        sid: {"id": sid, "fr": "a", "en_ref": "b", "terms": [{"word": "nope"}]},
    }
    results = [{"id": sid, "hyp": "c"}]
    v = compute_htm_hyp_vs_ref(results, _StubGraph(), id_to_segment)
    assert isinstance(v, float) and math.isnan(v)


def test_htm_vector_below_threshold_scores_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from pipeline.metrics import htm as htm_mod

    class _StubGraph:
        def ground(self, word: str, context=None):
            if word == "testspan":
                return {"name": "ExpectedEN", "level": 1}
            return None

        def candidate_renderings(self, _concept):
            return []

        def get_by_name(self, _name: str):
            return {"level": 1}

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    class _FakeModel:
        _list_encodes = 0

        def encode(self, x: object, convert_to_numpy: bool = True):
            if isinstance(x, str):
                return np.array([1.0, 0.0])
            lst = list(x)
            self._list_encodes += 1
            if self._list_encodes == 1:
                return np.stack([np.array([1.0, 0.0]) for _ in lst])
            return np.stack([np.array([0.0, 1.0]) for _ in lst])

    monkeypatch.setattr(htm_mod, "_get_sentence_encoder", lambda _name=None: _FakeModel())

    sid = "t1"
    id_to_segment = {
        sid: {
            "id": sid,
            "fr": "text with testspan inside",
            "terms": [{"word": "testspan"}],
        },
    }
    results = [{"id": sid, "hyp": "something orthogonal"}]
    assert (
        htm_mod.compute_htm_vector(
            results, _StubGraph(), id_to_segment, similarity_threshold=0.8
        )
        == 0.0
    )


def test_htm_vector_above_threshold_aligns(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from pipeline.metrics import htm as htm_mod

    class _StubGraph:
        def ground(self, word: str, context=None):
            if word == "testspan":
                return {"name": "ExpectedEN", "level": 2}
            return None

        def candidate_renderings(self, _concept):
            return []

        def get_by_name(self, name: str):
            if name == "ExpectedEN":
                return {"level": 2}
            return None

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    class _FakeModel:
        _list_encodes = 0

        def encode(self, x: object, convert_to_numpy: bool = True):
            if isinstance(x, str):
                return np.array([1.0, 0.0, 0.0])
            lst = list(x)
            self._list_encodes += 1
            vec = np.array([1.0, 0.0, 0.0])
            return np.stack([vec for _ in lst])

    monkeypatch.setattr(htm_mod, "_get_sentence_encoder", lambda _name=None: _FakeModel())

    sid = "t1"
    id_to_segment = {
        sid: {
            "id": sid,
            "fr": "text with testspan inside",
            "terms": [{"word": "testspan"}],
        },
    }
    results = [{"id": sid, "hyp": "blah"}]
    assert (
        htm_mod.compute_htm_vector(
            results, _StubGraph(), id_to_segment, similarity_threshold=0.8
        )
        == 1.0
    )


def test_macro_corpus_metric_by_group_is_mean_per_doc() -> None:
    from pipeline.metrics.corpus_scores import corpus_bleu, macro_corpus_metric_by_group

    # Two documents; perfect copies within each doc.
    hyps = ["alpha beta gamma", "delta epsilon", "zeta"]
    refs = list(hyps)
    groups = ["d0", "d0", "d1"]
    macro = macro_corpus_metric_by_group(corpus_bleu, hyps, refs, groups)
    b0 = corpus_bleu(hyps[:2], refs[:2])
    b1 = corpus_bleu(hyps[2:], refs[2:])
    assert macro == pytest.approx((b0 + b1) / 2.0)


def test_macro_corpus_metric_single_group_matches_corpus() -> None:
    from pipeline.metrics.corpus_scores import corpus_bleu, corpus_chrf, macro_corpus_metric_by_group

    hyps = ["the cat sat", "on the mat"]
    refs = ["a cat sat", "on my mat"]
    groups = ["x", "x"]
    assert macro_corpus_metric_by_group(corpus_bleu, hyps, refs, groups) == pytest.approx(
        corpus_bleu(hyps, refs)
    )
    assert macro_corpus_metric_by_group(corpus_chrf, hyps, refs, groups) == pytest.approx(
        corpus_chrf(hyps, refs)
    )


def test_macro_bleu_doc_concat_two_docs() -> None:
    from pipeline.metrics.corpus_scores import corpus_bleu, macro_bleu_doc_concat

    hyps = ["hello world", "foo", "bar baz", "qux"]
    refs = ["hello there", "foo", "bar see", "qux"]
    groups = ["d0", "d0", "d1", "d1"]
    got = macro_bleu_doc_concat(hyps, refs, groups)
    b0 = corpus_bleu(["hello world foo"], ["hello there foo"])
    b1 = corpus_bleu(["bar baz qux"], ["bar see qux"])
    assert got == pytest.approx((b0 + b1) / 2.0)


def test_macro_bleu_doc_concat_single_doc_matches_one_line_corpus() -> None:
    from pipeline.metrics.corpus_scores import corpus_bleu, macro_bleu_doc_concat

    hyps = ["the cat", "sat"]
    refs = ["a cat", "sat"]
    groups = ["x", "x"]
    assert macro_bleu_doc_concat(hyps, refs, groups) == pytest.approx(
        corpus_bleu(["the cat sat"], ["a cat sat"])
    )


def test_macro_bleu_doc_concat_mismatched_lengths_nan() -> None:
    from pipeline.metrics.corpus_scores import macro_bleu_doc_concat

    assert math.isnan(macro_bleu_doc_concat(["a"], ["b", "c"], ["x"]))
