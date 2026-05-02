"""Write one hypothesis JSONL row with perf_counter timing."""

from __future__ import annotations

import time
from typing import Any, Callable

from pipeline.systems.data_io import write_result_row


def write_timed_result(
    out_f,
    system: str,
    seg: dict[str, Any],
    hypothesis_fn: Callable[[], str],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    t0 = time.perf_counter()
    hyp = hypothesis_fn()
    inference_s = round(time.perf_counter() - t0, 4)
    write_result_row(out_f, system=system, seg=seg, hyp=hyp, inference_s=inference_s, extra=extra)
