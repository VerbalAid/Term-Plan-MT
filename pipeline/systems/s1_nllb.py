"""S1: NLLB-200 FR→EN per segment (no graph)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TextIO

import torch
from tqdm import tqdm

from pipeline.systems.data_io import iter_limited
from pipeline.systems.models import load_nllb, nllb_forced_bos_eng
from pipeline.systems.runtime import REPO_ROOT, ensure_repo_on_syspath
from pipeline.systems.timed_row import write_timed_result

ensure_repo_on_syspath()


def translate_segment(seg: dict[str, Any]) -> str:
    tok, model = load_nllb()
    tok.src_lang = "fra_Latn"
    device = next(model.parameters()).device
    inputs = tok(seg["fr"], return_tensors="pt").to(device)
    forced_bos = nllb_forced_bos_eng(tok)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            num_beams=5,
            max_new_tokens=256,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()


def run(
    segments_path: Path,
    out_f: TextIO,
    graph=None,
    locks: dict[str, str] | None = None,
    limit: int | None = None,
    skip_ids: set[str] | None = None,
    exclude_segment_ids: frozenset[str] | None = None,
) -> None:
    for seg in tqdm(iter_limited(segments_path, limit, exclude_segment_ids), desc="s1"):
        if skip_ids and seg["id"] in skip_ids:
            continue
        write_timed_result(out_f, "s1", seg, lambda: translate_segment(seg))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=Path, default=REPO_ROOT / "data" / "section48" / "segments_ner.jsonl")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "ad_hoc" / "s1.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        run(args.segments, f, limit=args.limit)
