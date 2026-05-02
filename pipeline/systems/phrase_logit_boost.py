"""Shared logits bias: boost every subword of preferred English phrases (any tokenizer)."""

from __future__ import annotations

from transformers import LogitsProcessor


class PhraseLogitBoost(LogitsProcessor):
    """Add a constant to logits of all tokens that appear in the tokenization of each phrase."""

    def __init__(self, tokenizer, phrases: list[str], boost: float = 1.75):
        self.boost = boost
        self.token_ids: set[int] = set()
        for p in phrases:
            enc = tokenizer(p, add_special_tokens=False)
            for tid in enc["input_ids"]:
                self.token_ids.add(int(tid))

    def __call__(self, input_ids, scores):
        for tid in self.token_ids:
            if 0 <= tid < scores.shape[-1]:
                scores[:, tid] += self.boost
        return scores
