#!/usr/bin/env python3
"""Shuffle Alpaca ontology JSONL into train / dev / test splits (reproducible seed).

Example::

    PYTHONPATH=. python tools/data/split_ontology_sft_jsonl.py \\
      --input data/ontology_ner_full_hierarchical_alpaca.jsonl \\
      --out-dir data

Writes ``ontology_ner_full_hierarchical_{train,val,test}.jsonl`` under ``--out-dir`` when
the input basename matches ``ontology_ner_full_hierarchical_alpaca.jsonl``; otherwise uses
``{stem}_train.jsonl`` etc.

Then train with explicit splits::

    PYTHONPATH=. python extras/experiments/french_medical_ner/biomistral_ner_finetune_unsloth.py \\
      --ontology-only \\
      --ontology-train-jsonl data/ontology_ner_full_hierarchical_alpaca_train.jsonl \\
      --ontology-val-jsonl data/ontology_ner_full_hierarchical_alpaca_val.jsonl \\
      --ontology-test-jsonl data/ontology_ner_full_hierarchical_alpaca_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source JSONL (one {\"text\": ...} per line).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory for train/val/test files (default: project data/).",
    )
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction (default 0.8).")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction (default 0.1).")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (default 42).")
    args = ap.parse_args()

    inp = args.input if args.input.is_absolute() else ROOT / args.input
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    tr = float(args.train_ratio)
    va = float(args.val_ratio)
    if tr <= 0 or va <= 0 or tr + va >= 1.0:
        raise SystemExit("Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")

    lines: list[str] = []
    with inp.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            obj = json.loads(s)
            if "text" not in obj:
                raise SystemExit(f"Line missing 'text': {inp}")
            lines.append(json.dumps({"text": obj["text"]}, ensure_ascii=False))

    n = len(lines)
    if n == 0:
        raise SystemExit("No rows in input.")

    rng = random.Random(int(args.seed))
    rng.shuffle(lines)

    n_train = max(1, int(n * tr))
    n_val = max(1, int(n * va))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val

    train_lines = lines[:n_train]
    val_lines = lines[n_train : n_train + n_val]
    test_lines = lines[n_train + n_val :]

    stem = inp.stem
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_dir / f"{stem}_train.jsonl"
    out_val = out_dir / f"{stem}_val.jsonl"
    out_test = out_dir / f"{stem}_test.jsonl"

    def write_many(path: Path, rows: list[str]) -> None:
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    write_many(out_train, train_lines)
    write_many(out_val, val_lines)
    write_many(out_test, test_lines)

    print(f"Total rows: {n}", file=sys.stderr)
    print(f"Train: {len(train_lines)} → {out_train}", file=sys.stderr)
    print(f"Val:   {len(val_lines)} → {out_val}", file=sys.stderr)
    print(f"Test:  {len(test_lines)} → {out_test}", file=sys.stderr)
    print(f"seed={args.seed}", file=sys.stderr)


if __name__ == "__main__":
    main()
