"""S3: Mistral GraphRAG (MedDRA lines in prompt)."""

from __future__ import annotations

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


def translate_segment(graph, seg: dict[str, Any], locks: dict[str, str] | None) -> str:
    lines = medra_lines_with_locks(graph, seg, locks)
    tok, model = load_mistral_4bit()
    fr_ctx = truncate_full_section_for_token_budget(seg["fr"], "", lines, tok)
    prompt = build_mistral_prompt(fr_ctx, lines)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    return strip_inst_echo(full)


def run(
    segments_path: Path,
    out_f: TextIO,
    graph=None,
    locks: dict[str, str] | None = None,
    limit: int | None = None,
    skip_ids: set[str] | None = None,
    exclude_segment_ids: frozenset[str] | None = None,
) -> None:
    with term_graph_session(graph) as g:
        for seg in tqdm(iter_limited(segments_path, limit, exclude_segment_ids), desc="s3"):
            if skip_ids and seg["id"] in skip_ids:
                continue
            write_timed_result(out_f, "s3", seg, lambda se=seg: translate_segment(g, se, locks))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=Path, default=REPO_ROOT / "data" / "section48" / "segments_ner.jsonl")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "ad_hoc" / "s3.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        run(args.segments, f, limit=args.limit)
