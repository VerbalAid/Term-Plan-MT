"""QUAERO BRAT sentence loader (DISO, CHEM, PROC) for Unsloth NER training."""

from __future__ import annotations

import re
from pathlib import Path

LABEL_NAMES = ["O", "B-DISO", "I-DISO", "B-CHEM", "I-CHEM", "B-PROC", "I-PROC"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for i, n in enumerate(LABEL_NAMES)}

KEEP_TYPES = {"DISO", "CHEM", "PROC"}

_WS_WORD = re.compile(r"\S+")


def _parse_brat_t_line(line: str) -> tuple[str, list[tuple[int, int]]] | None:
    """Return (entity_type, [(start,end), ...]) or None if not a T-entity line."""
    line = line.rstrip("\n")
    if not line.startswith("T") or "\t" not in line:
        return None
    parts = line.split("\t", 2)
    if len(parts) < 2:
        return None
    type_off = parts[1].strip()
    sp = type_off.find(" ")
    if sp == -1:
        return None
    entity_type = type_off[:sp].strip()
    offset_blob = type_off[sp + 1 :].strip()
    fragments: list[tuple[int, int]] = []
    for piece in offset_blob.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        nums = piece.split()
        if len(nums) >= 2:
            fragments.append((int(nums[0]), int(nums[1])))
    if not fragments:
        return None
    return entity_type, fragments


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or a0 >= b1)


def _resolve_overlaps(candidates: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Greedy: prefer longest contiguous span; drop shorter overlapping spans."""
    sorted_spans = sorted(candidates, key=lambda x: (x[1] - x[0]), reverse=True)
    accepted: list[tuple[int, int, str]] = []
    for s, e, t in sorted_spans:
        if e <= s:
            continue
        if any(_spans_overlap(s, e, a, b) for a, b, _ in accepted):
            continue
        accepted.append((s, e, t))
    return sorted(accepted, key=lambda x: (x[0], x[1]))


def _word_tags_from_spans(
    words: list[tuple[str, int, int]],
    accepted: list[tuple[int, int, str]],
) -> list[str]:
    """Assign BIO tags using token start: B if word starts at span start, else I."""
    tags: list[str] = []
    for _w, ws, we in words:
        lab = "O"
        for s, e, typ in accepted:
            if s <= ws < e:
                lab = f"B-{typ}" if ws == s else f"I-{typ}"
                break
        tags.append(lab)
    return tags


def _iter_line_spans(text: str):
    """Yield (line_slice, global_start, global_end_exclusive) for each newline-separated line."""
    start = 0
    n = len(text)
    while start <= n:
        nl = text.find("\n", start)
        if nl == -1:
            if start < n or (start == n and n == 0):
                yield text[start:], start, n
            break
        yield text[start:nl], start, nl
        start = nl + 1


def _parse_brat_raw_spans(text: str, ann_path: Path) -> list[tuple[int, int, str]]:
    raw_spans: list[tuple[int, int, str]] = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue
        parsed = _parse_brat_t_line(line)
        if parsed is None:
            continue
        ent_type, fragments = parsed
        if ent_type not in KEEP_TYPES:
            continue
        for s, e in fragments:
            if e <= s or s < 0 or e > len(text):
                continue
            raw_spans.append((s, e, ent_type))
    return raw_spans


def _spans_for_line(
    raw_spans: list[tuple[int, int, str]],
    g0: int,
    g1: int,
) -> list[tuple[int, int, str]]:
    """Clip global BRAT spans to [g0, g1) and express in line-local character offsets."""
    line_candidates: list[tuple[int, int, str]] = []
    for s, e, t in raw_spans:
        lo = max(s, g0)
        hi = min(e, g1)
        if hi <= lo:
            continue
        line_candidates.append((lo - g0, hi - g0, t))
    return line_candidates


def _sentence_records_from_brat_pair(txt_path: Path, ann_path: Path) -> list[dict[str, object]]:
    text = txt_path.read_text(encoding="utf-8")
    raw_spans = _parse_brat_raw_spans(text, ann_path)
    out: list[dict[str, object]] = []
    for _line_slice, g0, g1 in _iter_line_spans(text):
        segment = text[g0:g1]
        words: list[tuple[str, int, int]] = [
            (m.group(), m.start(), m.end()) for m in _WS_WORD.finditer(segment)
        ]
        if not words:
            continue
        line_candidates = _spans_for_line(raw_spans, g0, g1)
        accepted = _resolve_overlaps(line_candidates)
        tag_strs = _word_tags_from_spans(words, accepted)
        ner_ids = [LABEL2ID.get(ts, LABEL2ID["O"]) for ts in tag_strs]
        out.append(
            {
                "tokens": [w for w, _, _ in words],
                "ner_tags": ner_ids,
            }
        )
    return out


def load_quaero_brat(brat_dir: Path, *, combine_medline: bool = True) -> list[dict[str, object]]:
    """Read paired .txt / .ann under ``brat_dir`` (and optionally sibling MEDLINE).

    Each **line** of a ``.txt`` file is one training sentence. Spans are clipped to the line
    and overlap-resolved per sentence. After collecting all sentences from all files, the caller
    shuffles and splits train/val.
    """
    brat_dir = brat_dir.resolve()
    if not brat_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {brat_dir}")

    dirs_to_scan: list[Path] = [brat_dir]
    if combine_medline:
        med = brat_dir.parent / "MEDLINE"
        if med.is_dir():
            dirs_to_scan.append(med)

    records: list[dict[str, object]] = []
    for folder in dirs_to_scan:
        txt_files = sorted(folder.glob("*.txt"))
        for txt_path in txt_files:
            ann_path = txt_path.with_suffix(".ann")
            if not ann_path.is_file():
                continue
            records.extend(_sentence_records_from_brat_pair(txt_path, ann_path))

    if not records:
        searched = ", ".join(str(d) for d in dirs_to_scan)
        raise ValueError(f"No usable BRAT sentence examples under: {searched}")
    return records
