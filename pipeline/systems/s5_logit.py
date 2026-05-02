"""S5: NLLB or Mistral decoding with phrase logit boost."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, TextIO

import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

from pipeline.systems.data_io import iter_limited, write_result_row
from pipeline.systems.english_phrases import collect_english_phrases
from pipeline.systems.mistral_prompts import build_mistral_prompt, medra_lines_with_locks
from pipeline.systems.models import load_mistral_4bit, load_nllb, nllb_forced_bos_eng, strip_inst_echo
from pipeline.systems.phrase_logit_boost import PhraseLogitBoost
from pipeline.systems.runtime import REPO_ROOT, ensure_repo_on_syspath, term_graph_session

ensure_repo_on_syspath()

_LOG = logging.getLogger(__name__)


def _translate_nllb(graph, seg: dict[str, Any], locks: dict[str, str] | None) -> str:
    phrases = collect_english_phrases(graph, seg, locks)
    tok, model = load_nllb()
    tok.src_lang = "fra_Latn"
    device = next(model.parameters()).device
    inputs = tok(seg["fr"], return_tensors="pt").to(device)
    forced_bos = nllb_forced_bos_eng(tok)
    boost = PhraseLogitBoost(tok, phrases, boost=1.75)
    processors = LogitsProcessorList([boost])

    phrase_token_lists: list[list[int]] = []
    for p in phrases:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if ids:
            phrase_token_lists.append(ids)

    beams = min(10, max(5, 5 + len(phrase_token_lists)))
    gen_kw: dict[str, Any] = {
        "forced_bos_token_id": forced_bos,
        "logits_processor": processors,
        "num_beams": beams,
        "max_new_tokens": 256,
    }

    if phrase_token_lists:
        attempt = dict(gen_kw)
        attempt["force_words_ids"] = phrase_token_lists
        try:
            with torch.inference_mode():
                out = model.generate(**inputs, **attempt)
            return tok.decode(out[0], skip_special_tokens=True).strip()
        except (TypeError, ValueError, RuntimeError) as e:
            _LOG.debug("NLLB constrained beam skipped (%s); soft boost only.", e)

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kw)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _translate_mistral(graph, seg: dict[str, Any], locks: dict[str, str] | None) -> str:
    phrases = collect_english_phrases(graph, seg, locks)
    lines = medra_lines_with_locks(graph, seg, locks)
    prompt = build_mistral_prompt(seg["fr"], lines)
    tok, model = load_mistral_4bit()
    boost = PhraseLogitBoost(tok, phrases, boost=1.25)
    processors = LogitsProcessorList([boost])
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tok.eos_token_id,
            logits_processor=processors,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    return strip_inst_echo(full)


def translate_segment(
    graph,
    seg: dict[str, Any],
    locks: dict[str, str] | None,
    backend: str,
) -> tuple[str, str]:
    b = backend.lower().strip()
    if b == "mistral":
        return _translate_mistral(graph, seg, locks), "mistral"
    return _translate_nllb(graph, seg, locks), "nllb"


def run(
    segments_path: Path,
    out_f: TextIO,
    graph=None,
    locks: dict[str, str] | None = None,
    limit: int | None = None,
    s5_backend: str | None = None,
    skip_ids: set[str] | None = None,
    exclude_segment_ids: frozenset[str] | None = None,
) -> None:
    backend = (s5_backend or os.environ.get("S5_BACKEND", "nllb")).lower().strip()

    with term_graph_session(graph) as g:
        for seg in tqdm(iter_limited(segments_path, limit, exclude_segment_ids), desc=f"s5[{backend}]"):
            if skip_ids and seg["id"] in skip_ids:
                continue
            t0 = time.perf_counter()
            hyp, dec = translate_segment(g, seg, locks, backend)
            inference_s = round(time.perf_counter() - t0, 4)
            write_result_row(
                out_f,
                system="s5",
                seg=seg,
                hyp=hyp,
                inference_s=inference_s,
                extra={"decoder": dec},
            )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=Path, default=REPO_ROOT / "data" / "section48" / "segments_ner.jsonl")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "ad_hoc" / "s5.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--s5-backend", choices=["nllb", "mistral"], default="nllb")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        run(args.segments, f, limit=args.limit, s5_backend=args.s5_backend)
