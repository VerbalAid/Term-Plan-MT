"""One-shot COMET corpus score (subprocess entrypoint for evaluate.py).

PyTorch Lightning + CUDA can fail on repeated ``predict`` in the same process; ``evaluate.py``
invokes this module once per system so each run owns the GPU lifecycle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.metrics.comet_once <path.json>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        sys.exit(1)
    try:
        import torch
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        print(f"import_error: {e}", file=sys.stderr)
        sys.exit(1)
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    use_gpu = 1 if torch.cuda.is_available() else 0
    bs = 16 if use_gpu else 8
    out = model.predict(rows, batch_size=bs, gpus=use_gpu)
    print(float(out.system_score), flush=True)


if __name__ == "__main__":
    main()
