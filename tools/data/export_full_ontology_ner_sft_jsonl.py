#!/usr/bin/env python3
"""Build Alpaca JSONL from **every** Neo4j ``:Concept`` with a French ``fr_label``.

One training line per retained concept: ``### Input`` is that French label (short wrapper text);
``### Response`` is a JSON list of one object. **legacy** schema matches early exports
``{fr, en, level, tier, id}``. **hierarchical** (default) adds SOC→LLT path fields, ``en_resolved``,
and PT-canonical ``en`` (schema documented in ``pipeline/ontology_sft_alpaca.py``).

Use ``--prompt-style mistral`` for Mistral-7B-Instruct ``[INST]…[/INST]`` framing (recommended
for that base model). Default ``alpaca`` keeps ``### Instruction`` / ``### Response`` blocks.

``biomistral_ner_finetune_unsloth.py --ontology-sft-jsonl`` accepts all of these; training only
uses ``fr`` from the gold JSON for the built-in multiset F1, so extra keys are learnable structure.

French labels that normalize to the same key but map to **different** MedDRA ids are ambiguous
for SFT (identical input, different gold). By default we keep a **single representative** per
normalized French key (prefer PT, then lowest ``level``, then ``id``). Use ``--fr-dedupe`` to
change that.

Requires Neo4j (same env as ``TermGraph``).

Example::

    PYTHONPATH=. python tools/data/export_full_ontology_ner_sft_jsonl.py \\
      --out data/ontology_ner_full_hierarchical_alpaca.jsonl

    PYTHONPATH=. python extras/experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py \\
      --ontology-sft-jsonl data/ontology_ner_full_hierarchical_alpaca.jsonl --ontology-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.graph import TermGraph, normalize_fr_for_grounding
from pipeline.ontology_sft_alpaca import (
    row_payload_json,
    row_payload_json_hierarchical,
    to_alpaca,
    to_alpaca_hierarchical,
    to_mistral_instruct,
    to_mistral_instruct_hierarchical,
)


def _pick_representative(group: list[dict[str, Any]]) -> dict[str, Any]:
    def sk(r: dict[str, Any]) -> tuple:
        tier = str(r.get("tier") or "").upper()
        is_pt = 0 if tier == "PT" else 1
        lvl = r.get("level")
        try:
            lv = int(lvl)
        except (TypeError, ValueError):
            lv = 999
        return (is_pt, lv, str(r.get("id") or ""))

    return sorted(group, key=sk)[0]


def _apply_fr_dedupe(
    rows: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return (filtered_rows, n_colliding_keys, n_dropped_rows)."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        fl = (r.get("fr_label") or "").strip()
        if not fl:
            continue
        buckets[normalize_fr_for_grounding(fl)].append(r)

    out: list[dict[str, Any]] = []
    n_colliding = 0
    n_dropped = 0
    for _key, group in buckets.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        n_colliding += 1
        if mode == "none":
            out.extend(group)
        elif mode == "skip":
            n_dropped += len(group)
            continue
        else:
            chosen = _pick_representative(group)
            out.append(chosen)
            n_dropped += len(group) - 1
    return out, n_colliding, n_dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL (one {\"text\": ...} per retained concept). "
        "Default: data/ontology_ner_full_hierarchical_alpaca.jsonl or _legacy_ if --format legacy.",
    )
    ap.add_argument(
        "--format",
        choices=("legacy", "hierarchical"),
        default="hierarchical",
        help="hierarchical = PT-canonical en + path (default); legacy = original 5-key rows.",
    )
    ap.add_argument(
        "--supervision-en",
        choices=("pt", "grounded"),
        default="pt",
        help="hierarchical only: set primary `en` to PT (pt) or grounded node (grounded).",
    )
    ap.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        default=None,
        metavar="TIER",
        help="Restrict to MedDRA tier(s), e.g. ``--tier PT --tier LLT``. Default: all tiers with fr_label.",
    )
    ap.add_argument(
        "--fr-dedupe",
        choices=("representative", "none", "skip"),
        default="representative",
        help="How to handle multiple concepts sharing the same normalized fr_label: "
        "``representative`` (default) = one row per FR key; ``none`` = one row per concept (can duplicate inputs); "
        "``skip`` = drop every concept in a colliding FR key.",
    )
    ap.add_argument(
        "--prompt-style",
        choices=("alpaca", "mistral"),
        default="alpaca",
        help="alpaca = ### Instruction/Input/Response (default). mistral = <s>[INST]…[/INST]…</s>.",
    )
    args = ap.parse_args()

    if args.out is None:
        base = (
            "ontology_ner_full_hierarchical_alpaca.jsonl"
            if args.format == "hierarchical"
            else "ontology_ner_full_legacy_alpaca.jsonl"
        )
        out_path = ROOT / "data" / base
    else:
        out_path = args.out if args.out.is_absolute() else ROOT / args.out
    tiers = list(args.tiers) if args.tiers else None

    graph = TermGraph(grounding_mode="string")
    try:
        raw_rows = graph.fetch_concepts_with_fr_labels(tiers=tiers)
        if args.format == "hierarchical":
            graph.preload_hierarchy_index()
        kept, n_colliding, n_dropped = _apply_fr_dedupe(raw_rows, mode=args.fr_dedupe)

        n_out = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fout:
            for r in kept:
                fr_raw = (r.get("fr_label") or "").strip()
                if not fr_raw:
                    continue
                concept = {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "level": r.get("level"),
                    "tier": r.get("tier"),
                }
                en = str(concept.get("name") or "").strip()
                if not en or concept.get("level") is None:
                    continue
                if args.format == "hierarchical":
                    cid = str(concept.get("id") or "").strip()
                    if not cid:
                        continue
                    hier = graph.fetch_hierarchy_for_concept(cid)
                    payload = row_payload_json_hierarchical(
                        concept,
                        hier,
                        fr_surface=fr_raw,
                        supervision_en=args.supervision_en,
                    )
                    wrap = (
                        to_mistral_instruct_hierarchical
                        if args.prompt_style == "mistral"
                        else to_alpaca_hierarchical
                    )
                else:
                    payload = row_payload_json(concept, fr_surface=fr_raw)
                    wrap = to_mistral_instruct if args.prompt_style == "mistral" else to_alpaca
                body = f"Texte (libellé médical en français) :\n{fr_raw}"
                text = wrap(body, payload)
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n_out += 1
    finally:
        graph.close()

    print(f"Neo4j concepts with fr_label (before FR-key policy): {len(raw_rows)}", file=sys.stderr)
    print(f"Normalized-FR keys with >1 concept: {n_colliding}", file=sys.stderr)
    print(f"Rows dropped by FR policy: {n_dropped}", file=sys.stderr)
    print(
        f"Wrote {n_out} JSONL rows → {out_path} (format={args.format})",
        file=sys.stderr,
    )
    if args.format == "hierarchical":
        print(f"supervision_en={args.supervision_en}", file=sys.stderr)
    print(f"prompt_style={args.prompt_style}", file=sys.stderr)


if __name__ == "__main__":
    main()
