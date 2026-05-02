"""Read aligned segments from JSONL; write standard result rows."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO


def parse_exclude_segment_ids(spec: str | None) -> frozenset[str]:
    """Comma-separated segment ids (e.g. ``48_028`` for Section 4.8 Table 2 block)."""
    if not spec or not str(spec).strip():
        return frozenset()
    return frozenset(x.strip() for x in str(spec).split(",") if x.strip())


def iter_segments(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_segments_filtered(
    path: Path,
    exclude_segment_ids: frozenset[str] | None = None,
) -> Iterator[dict[str, Any]]:
    for seg in iter_segments(path):
        sid = seg.get("id")
        if exclude_segment_ids and sid in exclude_segment_ids:
            continue
        yield seg


def iter_limited(
    path: Path,
    limit: int | None,
    exclude_segment_ids: frozenset[str] | None = None,
) -> Iterator[dict[str, Any]]:
    for i, seg in enumerate(iter_segments_filtered(path, exclude_segment_ids)):
        if limit is not None and i >= limit:
            break
        yield seg


def load_all_segments(
    path: Path,
    exclude_segment_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    return list(iter_segments_filtered(path, exclude_segment_ids))


def write_result_row(
    out_f: TextIO,
    *,
    system: str,
    seg: dict[str, Any],
    hyp: str,
    inference_s: float,
    extra: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "id": seg["id"],
        "system": system,
        "fr": seg["fr"],
        "hyp": hyp,
        "en_ref": seg["en_ref"],
        "inference_s": inference_s,
    }
    if extra:
        row.update(extra)
    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_f.flush()
