#!/usr/bin/env python3
"""Count sentence-level lines in QUAERO BRAT train folders (EMEA / MEDLINE)."""

from __future__ import annotations

import argparse
from pathlib import Path


def count_lines(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    total = 0
    for txt in sorted(folder.glob("*.txt")):
        text = txt.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                total += 1
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emea", type=Path, help="Path to corpus/train/EMEA")
    p.add_argument("--medline", type=Path, help="Path to corpus/train/MEDLINE")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("data/QUAERO_FrenchMed/corpus/train"),
        help="If set, use <root>/EMEA and <root>/MEDLINE when --emea/--medline omitted",
    )
    args = p.parse_args()

    emea = args.emea or (args.root / "EMEA")
    medline = args.medline or (args.root / "MEDLINE")
    n_emea = count_lines(emea)
    n_med = count_lines(medline)
    n_total = n_emea + n_med

    print(f"EMEA:    {n_emea:5d}  ({emea})")
    print(f"MEDLINE: {n_med:5d}  ({medline})")
    print(f"TOTAL:   {n_total:5d}  sentence-level lines")
    if n_total:
        print(f"90/10 split (typical): ~{int(n_total * 0.9)} train · ~{n_total - int(n_total * 0.9)} val")


if __name__ == "__main__":
    main()
