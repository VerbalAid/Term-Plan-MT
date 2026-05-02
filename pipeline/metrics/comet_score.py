"""Corpus-level COMET-DA via one isolated subprocess per call (CUDA / Lightning safe)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def corpus_comet_da(srcs: list[str], hyps: list[str], refs: list[str]) -> float | None:
    if not srcs or len(srcs) != len(hyps) or len(hyps) != len(refs):
        return None
    rows = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"rows": rows}, f)
        proc = subprocess.run(
            [sys.executable, "-m", "pipeline.metrics.comet_once", tmp],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        if not out:
            return None
        return float(out.splitlines()[-1].strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return None
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
