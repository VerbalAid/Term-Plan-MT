"""JSONL loading and hypothesis/reference alignment (evaluate + plotting)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results_jsonl(path: Path, partial: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if partial:
                        continue
                    raise
    return rows


def align_refs(results: list[dict], id_to_ref: dict[str, str]) -> tuple[list[str], list[str]]:
    hyps: list[str] = []
    refs: list[str] = []
    for r in sorted(results, key=lambda x: x["id"]):
        rid = r["id"]
        if rid not in id_to_ref:
            continue
        hyps.append(r.get("hyp", ""))
        refs.append(id_to_ref[rid])
    return hyps, refs


def align_src_hyp_ref(
    results: list[dict],
    id_to_ref: dict[str, str],
    id_to_src: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    srcs: list[str] = []
    hyps: list[str] = []
    refs: list[str] = []
    for r in sorted(results, key=lambda x: x["id"]):
        rid = r["id"]
        if rid not in id_to_ref or rid not in id_to_src:
            continue
        srcs.append(id_to_src[rid])
        hyps.append(r.get("hyp", ""))
        refs.append(id_to_ref[rid])
    return srcs, hyps, refs
