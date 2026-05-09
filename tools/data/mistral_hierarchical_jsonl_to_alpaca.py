#!/usr/bin/env python3
"""Rewrite hierarchical ontology JSONL from Mistral ``[INST]…[/INST]`` to Alpaca blocks.

Each line is ``{"text": "…"}``. Only the framing changes; the JSON list in the response
is copied verbatim (so it stays mdhier-patched if the Mistral file was patched).

Example::

    PYTHONPATH=. python tools/data/mistral_hierarchical_jsonl_to_alpaca.py \\
      --input data/ontology_ner_full_hierarchical_mistral_train.jsonl \\
      --output data/ontology_ner_full_hierarchical_alpaca.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.ontology_sft_alpaca import to_alpaca_hierarchical

_MISTRAL_PREFIX = "<s>[INST] "
_MISTRAL_SEP = " [/INST] "


def split_mistral_hierarchical_text(text: str) -> tuple[str, str]:
    if not text.startswith(_MISTRAL_PREFIX):
        raise ValueError("expected Mistral line starting with <s>[INST] ")
    idx = text.find(_MISTRAL_SEP)
    if idx < 0:
        raise ValueError("missing [/INST] delimiter")
    user = text[len(_MISTRAL_PREFIX) : idx]
    assistant = text[idx + len(_MISTRAL_SEP) :].rstrip()
    if assistant.endswith("</s>"):
        assistant = assistant[: -len("</s>")].rstrip()
    marker = "### Input:\n"
    mi = user.find(marker)
    if mi < 0:
        raise ValueError("missing ### Input:\\n in user turn")
    fr_body = user[mi + len(marker) :].strip()
    return fr_body, assistant


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    inp = args.input if args.input.is_absolute() else ROOT / args.input
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    n_ok = 0
    out_lines: list[str] = []
    with inp.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            t = row.get("text")
            if not isinstance(t, str):
                raise SystemExit(f"line {lineno}: missing text")
            fr_body, assistant = split_mistral_hierarchical_text(t)
            row["text"] = to_alpaca_hierarchical(fr_body, assistant)
            out_lines.append(json.dumps(row, ensure_ascii=False))
            n_ok += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {n_ok} lines -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
