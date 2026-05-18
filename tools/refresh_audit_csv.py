#!/usr/bin/env python3
"""Re-fill audit_annotated.csv text columns from JSONL sources (keeps annotations)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEGS_PATH = ROOT / "data/section48/segments_ner_biollm.jsonl"
S1_PATH = ROOT / "results/ner_biollm/s1.jsonl"
S2_PATH = ROOT / "results/ner_biollm/s2.jsonl"
S5_PATH = ROOT / "results/ner_biollm/s5_mistral.jsonl"
S6_PATH = ROOT / "results/ner_biollm/s6.jsonl"
AUDIT_CSV = ROOT / "error_analysis/annotations/audit_annotated.csv"
EXCLUDE_SEGMENT_IDS = frozenset({"48_028"})


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    if not AUDIT_CSV.is_file():
        print(f"Missing {AUDIT_CSV}", file=sys.stderr)
        sys.exit(1)

    segs = {r["id"]: r for r in load_jsonl(SEGS_PATH)}
    hyps = {
        "s1": {r["id"]: r.get("hyp", "") for r in load_jsonl(S1_PATH)},
        "s2": {r["id"]: r.get("hyp", "") for r in load_jsonl(S2_PATH)},
        "s5": {r["id"]: r.get("hyp", "") for r in load_jsonl(S5_PATH)},
        "s6": {r["id"]: r.get("hyp", "") for r in load_jsonl(S6_PATH)},
    }

    with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No rows in audit CSV", file=sys.stderr)
        sys.exit(1)

    rows = [r for r in rows if r["id"] not in EXCLUDE_SEGMENT_IDS]

    for row in rows:
        sid = row["id"]
        if sid not in segs:
            print(f"Warning: segment {sid} not in {SEGS_PATH}", file=sys.stderr)
            continue
        seg = segs[sid]
        row["fr"] = seg.get("fr", "")
        row["human_ref"] = seg.get("en_ref", "")
        for key in hyps:
            row[key] = hyps[key].get(sid, "")

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {len(rows)} rows → {AUDIT_CSV}")


if __name__ == "__main__":
    main()
