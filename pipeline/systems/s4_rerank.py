"""S4: Mistral GraphRAG + sample + ontology-aware rerank."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TextIO

import torch
from tqdm import tqdm

from pipeline.systems.data_io import iter_limited
from pipeline.systems.mistral_prompts import (
    build_mistral_prompt,
    medra_lines_with_locks,
    truncate_full_section_for_token_budget,
)
from pipeline.systems.models import load_mistral_4bit, strip_inst_echo
from pipeline.systems.runtime import REPO_ROOT, ensure_repo_on_syspath, term_graph_session
from pipeline.systems.timed_row import write_timed_result

ensure_repo_on_syspath()

S4_DEFAULT_NUM_CANDIDATES = 3
S4_MAX_NUM_CANDIDATES = 5


def _names_in_text(names: list[str], text: str) -> list[str]:
    tl = text.lower()
    found: list[str] = []
    for name in names:
        if name and name.lower() in tl:
            found.append(name)
    return found


def _best_score_for_term(graph, concept: dict, candidate: str, names: list[str]) -> float:
    pref = concept["name"]
    cl = candidate.lower()
    if pref.lower() in cl:
        return 1.0
    best = 0.0
    for found in _names_in_text(names, candidate):
        if found.lower() == pref.lower():
            return 1.0
        fn = graph.get_by_name(found)
        if not fn:
            continue
        if fn["level"] == concept["level"]:
            best = max(best, 1.0)
        elif graph.same_branch(found, pref):
            best = max(best, 0.5)
    return best


def score_candidate(graph, seg: dict[str, Any], candidate: str, names: list[str]) -> float | None:
    score = 0.0
    count = 0
    fr_ctx = (seg.get("fr") or "").strip() or None
    for term in seg.get("terms") or []:
        concept = graph.ground(term.get("word", ""), context=fr_ctx)
        if not concept:
            continue
        score += _best_score_for_term(graph, concept, candidate, names)
        count += 1
    return score / count if count > 0 else None


def translate_segment(
    graph,
    seg: dict[str, Any],
    names: list[str],
    locks: dict[str, str] | None,
    *,
    num_candidates: int | None = None,
) -> str:
    lines = medra_lines_with_locks(graph, seg, locks)
    tok, model = load_mistral_4bit()
    fr_ctx = truncate_full_section_for_token_budget(seg["fr"], "", lines, tok)
    prompt = build_mistral_prompt(fr_ctx, lines)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if num_candidates is None:
        n_samples = int(os.environ.get("S4_NUM_SAMPLES", str(S4_DEFAULT_NUM_CANDIDATES)))
    else:
        n_samples = num_candidates
    n_samples = max(1, min(S4_MAX_NUM_CANDIDATES, n_samples))
    max_new = int(os.environ.get("S4_MAX_NEW_TOKENS", "512"))
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max_new,
        "pad_token_id": tok.eos_token_id,
        "do_sample": True,
        "temperature": 0.2,
    }
    parallel = os.environ.get("S4_PARALLEL_GENERATE", "").strip() == "1"
    cands: list[str] = []
    with torch.inference_mode():
        if n_samples == 1 or parallel:
            kw = {**gen_kw, "num_return_sequences": n_samples}
            out = model.generate(**inputs, **kw)
            for i in range(n_samples):
                full = tok.decode(out[i], skip_special_tokens=True)
                cands.append(strip_inst_echo(full))
            del out
        else:
            kw1 = {**gen_kw, "num_return_sequences": 1}
            for _ in range(n_samples):
                out = model.generate(**inputs, **kw1)
                full = tok.decode(out[0], skip_special_tokens=True)
                cands.append(strip_inst_echo(full))
                del out
    best = cands[0]
    best_s = score_candidate(graph, seg, best, names)
    for c in cands[1:]:
        s = score_candidate(graph, seg, c, names)
        if s is not None and (best_s is None or s > best_s):
            best, best_s = c, s
    if best_s is None:
        logging.debug("S4 rerank skipped: no grounded spans for %s", seg["id"])
        return cands[0]
    return best


def run(
    segments_path: Path,
    out_f: TextIO,
    graph=None,
    locks: dict[str, str] | None = None,
    limit: int | None = None,
    skip_ids: set[str] | None = None,
    *,
    s4_candidates: int | None = None,
    exclude_segment_ids: frozenset[str] | None = None,
) -> None:
    with term_graph_session(graph) as g:
        names = g.list_concept_names()
        for seg in tqdm(iter_limited(segments_path, limit, exclude_segment_ids), desc="s4"):
            if skip_ids and seg["id"] in skip_ids:
                continue
            write_timed_result(
                out_f,
                "s4",
                seg,
                lambda se=seg: translate_segment(g, se, names, locks, num_candidates=s4_candidates),
            )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=Path, default=REPO_ROOT / "data" / "section48" / "segments_ner.jsonl")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "ad_hoc" / "s4.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        run(args.segments, f, limit=args.limit)
