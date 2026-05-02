"""Unit tests for HTM and matching helpers."""

from __future__ import annotations

import pytest

from pipeline.metrics.htm import compute_htm
from pipeline.metrics.matching import normalize_text, phrase_in_hyp


def test_htm_miss_scores_zero() -> None:
    class _StubGraph:
        def get_by_name(self, _name: str):
            return None

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    gold = [{"fr": "goldspan", "en_label": "ExpectedEN", "en_aliases": [], "level": 1}]
    results = [{"fr": "text with goldspan inside", "hyp": "no matching english phrase here"}]
    assert compute_htm(results, gold, _StubGraph()) == 0.0


def test_normalize_hyphen_variants() -> None:
    a = normalize_text("immune-mediated")
    b = normalize_text("immune–mediated")
    c = normalize_text("immune mediated")
    assert a == b == c


def test_phrase_in_hyp_case_insensitive() -> None:
    assert phrase_in_hyp("Patient had PNEUMONIA", "pneumonia") is True
    assert phrase_in_hyp("patient had pneumonia", "PNEUMONIA") is True


def test_htm_vector_below_threshold_scores_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from pipeline.metrics import htm as htm_mod

    class _StubGraph:
        def get_by_name(self, name: str):
            return {"level": 1} if name else None

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    class _FakeModel:
        _call = 0

        def encode(self, x: object, convert_to_numpy: bool = True):
            _FakeModel._call += 1
            if isinstance(x, str):
                return np.array([1.0, 0.0])
            lst = list(x)
            # First encode per segment row: hypothesis chunks → orthogonal to labels
            if _FakeModel._call == 1:
                return np.stack([np.array([1.0, 0.0]) for _ in lst])
            # Gold English renderings
            return np.stack([np.array([0.0, 1.0]) for _ in lst])

    monkeypatch.setattr(htm_mod, "_get_sentence_encoder", lambda _name=None: _FakeModel())

    gold = [{"fr": "goldspan", "en_label": "ExpectedEN", "en_aliases": [], "level": 1}]
    results = [{"fr": "text with goldspan inside", "hyp": "something orthogonal"}]
    assert (
        htm_mod.compute_htm_vector(results, gold, _StubGraph(), similarity_threshold=0.8) == 0.0
    )


def test_htm_vector_above_threshold_aligns(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from pipeline.metrics import htm as htm_mod

    class _StubGraph:
        def get_by_name(self, name: str):
            if name == "ExpectedEN":
                return {"level": 2}
            return None

        def same_branch(self, _a: str, _b: str) -> bool:
            return False

    class _FakeModel:
        _call = 0

        def encode(self, x: object, convert_to_numpy: bool = True):
            _FakeModel._call += 1
            if isinstance(x, str):
                return np.array([1.0, 0.0, 0.0])
            lst = list(x)
            if _FakeModel._call == 1:
                return np.stack([np.array([1.0, 0.0, 0.0]) for _ in lst])
            return np.stack([np.array([1.0, 0.0, 0.0]) for _ in lst])

    monkeypatch.setattr(htm_mod, "_get_sentence_encoder", lambda _name=None: _FakeModel())

    gold = [{"fr": "goldspan", "en_label": "ExpectedEN", "en_aliases": [], "level": 2}]
    results = [{"fr": "text with goldspan inside", "hyp": "blah"}]
    assert (
        htm_mod.compute_htm_vector(results, gold, _StubGraph(), similarity_threshold=0.8) == 1.0
    )
