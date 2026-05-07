#!/usr/bin/env python3
"""Propose FR→EN term rows from parallel FR / ``en_ref`` NER + position pairing on segment JSONL.

Pairs French and English entities in the same segment by midpoint proximity, drops nested
spans so short fragments are not glued to longer mentions, dedupes by a normalised French
key, then merges English variants as ``en_aliases``. Intended for **optional Neo4j seeding**
(`data/gold_terms.json` → `build_graph.py`) or human review — **not** consumed by HTM scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _norm_key(s: str) -> str:
    t = unicodedata.normalize("NFKC", (s or "").strip()).casefold()
    return " ".join(t.split())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _ner_entities(
    pipe: Any,
    text: str,
    *,
    min_score: float,
) -> list[dict[str, Any]]:
    raw = pipe(text or "")
    out: list[dict[str, Any]] = []
    for ent in raw:
        sc = float(ent.get("score", 0.0))
        if sc <= min_score:
            continue
        word = (ent.get("word") or "").strip()
        if len(word) < 2:
            continue
        start = int(ent.get("start", 0))
        end = int(ent.get("end", 0))
        if end <= start:
            continue
        out.append(
            {
                "word": word,
                "start": start,
                "end": end,
                "score": sc,
                "entity_group": str(ent.get("entity_group", ent.get("entity", ""))),
            }
        )
    return out


def _ner_entities_by_lines(
    pipe: Any,
    full_text: str,
    *,
    min_score: float,
    max_chunk_chars: int,
) -> list[dict[str, Any]]:
    """Run NER on each non-empty line (global offsets) to avoid one long truncation."""
    out: list[dict[str, Any]] = []
    base = 0
    for raw in (full_text or "").split("\n"):
        leading = len(raw) - len(raw.lstrip(" \t"))
        line = raw.strip()
        line_start = base + leading
        base += len(raw) + 1
        if len(line) < 3:
            continue
        chunk = line[:max_chunk_chars]
        for ent in _ner_entities(pipe, chunk, min_score=min_score):
            e2 = dict(ent)
            e2["start"] = int(e2["start"]) + line_start
            e2["end"] = int(e2["end"]) + line_start
            out.append(e2)
    return out


def _drop_nested(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep longer spans; drop any span strictly contained in another kept span."""
    spans = sorted(spans, key=lambda s: (s["end"] - s["start"], s["score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for s in spans:
        inside = False
        for k in kept:
            if s["start"] >= k["start"] and s["end"] <= k["end"] and (s["start"], s["end"]) != (
                k["start"],
                k["end"],
            ):
                inside = True
                break
        if not inside:
            kept.append(s)
    return kept


def _mid_char(s: dict[str, Any]) -> float:
    return (s["start"] + s["end"]) / 2.0


def _pair_by_position(
    fr_spans: list[dict[str, Any]],
    en_spans: list[dict[str, Any]],
    fr_text: str,
    en_text: str,
    *,
    max_char_dist: float,
) -> list[tuple[int, int]]:
    """Greedy 1:1 pairing by scaled midpoints; returns (fr_index, en_index)."""
    if not fr_spans or not en_spans:
        return []
    fr_len = max(len(fr_text), 1)
    en_len = max(len(en_text), 1)
    scale = en_len / fr_len

    order_fr = sorted(range(len(fr_spans)), key=lambda i: fr_spans[i]["start"])
    used_en: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i_fr in order_fr:
        fe = fr_spans[i_fr]
        fr_mid = _mid_char(fe)
        proj = fr_mid * scale
        best_j: int | None = None
        best_d = 1e9
        for j, ee in enumerate(en_spans):
            if j in used_en:
                continue
            d = abs(proj - _mid_char(ee))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= max_char_dist:
            used_en.add(best_j)
            pairs.append((i_fr, best_j))
    return pairs


def _pair_fuzzy_residual(
    fr_spans: list[dict[str, Any]],
    en_spans: list[dict[str, Any]],
    used_fr: set[int],
    used_en: set[int],
    *,
    min_ratio: int,
) -> list[tuple[int, int]]:
    """Greedy many-to-one avoided: sort (score, i, j) descending, take disjoint pairs."""
    from rapidfuzz import fuzz

    cand: list[tuple[int, int, int]] = []
    for i, fe in enumerate(fr_spans):
        if i in used_fr:
            continue
        fw = (fe.get("word") or "").strip()
        if len(fw) < 3:
            continue
        for j, ee in enumerate(en_spans):
            if j in used_en:
                continue
            ew = (ee.get("word") or "").strip()
            if len(ew) < 3:
                continue
            r = int(fuzz.token_sort_ratio(fw, ew))
            if r >= min_ratio:
                cand.append((r, i, j))
    cand.sort(key=lambda t: t[0], reverse=True)
    out: list[tuple[int, int]] = []
    seen_fr = set(used_fr)
    seen_en = set(used_en)
    for _r, i, j in cand:
        if i in seen_fr or j in seen_en:
            continue
        seen_fr.add(i)
        seen_en.add(j)
        out.append((i, j))
    return out


def _entity_allowed(label: str, allow: frozenset[str] | None) -> bool:
    if allow is None:
        return True
    return label.upper() in allow or label in allow


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Propose FR→EN term JSON rows from parallel FR/EN NER on segment JSONL."
    )
    ap.add_argument(
        "--segments",
        type=Path,
        default=ROOT / "data" / "section48" / "segments_ner_biollm.jsonl",
        help="JSONL with fr + en_ref per line.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON list path.",
    )
    ap.add_argument(
        "--merge-json",
        type=Path,
        default=None,
        help="Optional existing JSON to merge in (hand-crafted rows kept on clash).",
    )
    ap.add_argument(
        "--fr-model",
        type=str,
        default="Jean-Baptiste/camembert-ner",
        help="HF model id for French token-classification NER.",
    )
    ap.add_argument(
        "--en-model",
        type=str,
        default="dslim/bert-base-NER",
        help="HF model id for English token-classification NER.",
    )
    ap.add_argument("--min-score", type=float, default=0.72, help="Drop NER spans at or below this score.")
    ap.add_argument(
        "--max-char-dist",
        type=float,
        default=200.0,
        help="Max projected character distance for FR↔EN span pairing (scale maps FR mid to EN line).",
    )
    ap.add_argument(
        "--entity-groups",
        type=str,
        default="",
        help="Comma list of entity_group labels to keep (empty = keep all). Example: PER,MISC,ORG",
    )
    ap.add_argument(
        "--fuzzy-min-ratio",
        type=int,
        default=82,
        help="Second pass: pair leftover FR/EN spans when token_sort_ratio >= this (0–100). Use 0 to disable.",
    )
    ap.add_argument(
        "--single-chunk-ner",
        action="store_true",
        help="Run NER once on the full segment (default: per-line NER for better recall on long paragraphs).",
    )
    ap.add_argument(
        "--max-chunk-chars",
        type=int,
        default=512,
        help="Per-line mode: truncate each line to this many characters before NER.",
    )
    ap.add_argument(
        "--extra-segments",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Extra segment JSONLs (same keys: id, fr, en_ref) appended after --segments for more coverage.",
    )
    ap.add_argument("--limit-segments", type=int, default=None, help="Process only the first N rows after merge.")
    ap.add_argument("--max-rows", type=int, default=400, help="Cap output rows after merge (default 400).")
    args = ap.parse_args()

    import torch
    from transformers import pipeline

    seg_path = args.segments if args.segments.is_absolute() else ROOT / args.segments
    if not seg_path.is_file():
        raise SystemExit(f"segments file not found: {seg_path}")

    allow: frozenset[str] | None = None
    if args.entity_groups.strip():
        allow = frozenset(x.strip().upper() for x in args.entity_groups.split(",") if x.strip())

    device = 0 if torch.cuda.is_available() else -1
    ner_fr = pipeline("ner", model=args.fr_model, aggregation_strategy="max", device=device)
    ner_en = pipeline("ner", model=args.en_model, aggregation_strategy="max", device=device)

    rows = _load_jsonl(seg_path)
    for ep in args.extra_segments:
        ep = ep if ep.is_absolute() else (ROOT / ep)
        if ep.is_file():
            rows.extend(_load_jsonl(ep))
        else:
            print(f"WARN: extra segments file missing, skip: {ep}", file=sys.stderr)
    if args.limit_segments is not None:
        rows = rows[: max(0, args.limit_segments)]

    merged: dict[str, dict[str, Any]] = {}

    def _add_or_merge(fr_surface: str, en_surface: str) -> None:
        k = _norm_key(fr_surface)
        if not k:
            return
        en_surface = en_surface.strip()
        if not en_surface:
            return
        if k in merged:
            row = merged[k]
            el = (row.get("en_label") or "").strip()
            aliases = {x for x in (row.get("en_aliases") or []) if x}
            if el and _norm_key(en_surface) != _norm_key(el):
                aliases.add(en_surface)
            row["en_aliases"] = sorted(aliases, key=len)
            if len(fr_surface) > len(row.get("fr") or ""):
                row["fr"] = fr_surface.strip()
            return
        merged[k] = {
            "fr": fr_surface.strip(),
            "en_label": en_surface,
            "en_aliases": [],
        }

    if args.merge_json and args.merge_json.is_file():
        mj = args.merge_json if args.merge_json.is_absolute() else ROOT / args.merge_json
        prev = json.loads(mj.read_text(encoding="utf-8"))
        for g in prev:
            fr = (g.get("fr") or "").strip()
            en = (g.get("en_label") or "").strip()
            if fr and en:
                nk = _norm_key(fr)
                row = {
                    "fr": fr,
                    "en_label": en,
                    "en_aliases": list(g.get("en_aliases") or []),
                }
                for lk in ("level", "tier"):
                    if lk in g:
                        row[lk] = g[lk]
                merged[nk] = row

    n_seg = 0
    n_pairs = 0
    for row in rows:
        fr = (row.get("fr") or "").strip()
        en_ref = (row.get("en_ref") or "").strip()
        if not fr or not en_ref:
            continue
        if not args.single_chunk_ner:
            fr_e = _ner_entities_by_lines(ner_fr, fr, min_score=args.min_score, max_chunk_chars=args.max_chunk_chars)
            en_e = _ner_entities_by_lines(
                ner_en, en_ref, min_score=args.min_score, max_chunk_chars=args.max_chunk_chars
            )
        else:
            fr_e = _ner_entities(ner_fr, fr, min_score=args.min_score)
            en_e = _ner_entities(ner_en, en_ref, min_score=args.min_score)
        fr_e = [e for e in fr_e if _entity_allowed(e["entity_group"], allow)]
        en_e = [e for e in en_e if _entity_allowed(e["entity_group"], allow)]
        fr_e = _drop_nested(fr_e)
        en_e = _drop_nested(en_e)
        used_fr: set[int] = set()
        used_en: set[int] = set()
        idx_pairs = _pair_by_position(fr_e, en_e, fr, en_ref, max_char_dist=args.max_char_dist)
        for i, j in idx_pairs:
            used_fr.add(i)
            used_en.add(j)
        if args.fuzzy_min_ratio > 0:
            for i, j in _pair_fuzzy_residual(
                fr_e,
                en_e,
                used_fr,
                used_en,
                min_ratio=args.fuzzy_min_ratio,
            ):
                used_fr.add(i)
                used_en.add(j)
                idx_pairs.append((i, j))
        for i, j in idx_pairs:
            fr_w = fr_e[i]["word"].strip()
            en_w = en_e[j]["word"].strip()
            _add_or_merge(fr_w, en_w)
            n_pairs += 1
        n_seg += 1

    out_list = list(merged.values())
    out_list.sort(key=lambda r: _norm_key(r["fr"]))
    if len(out_list) > args.max_rows:
        out_list = out_list[: args.max_rows]

    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_list, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Segments processed: {n_seg}")
    print(f"Raw pairings (before dedupe): {n_pairs}")
    print(f"Unique French keys (rows): {len(merged)}")
    print(f"Wrote {len(out_list)} rows → {out_path}")


if __name__ == "__main__":
    main()
