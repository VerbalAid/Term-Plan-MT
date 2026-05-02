"""Wall-clock inference timing helpers (perf_counter)."""

from __future__ import annotations

import math
import statistics


def inference_mean_p95(rows: list[dict]) -> tuple[float | None, float | None]:
    """Mean and discrete 95th percentile of `inference_s` fields (seconds per segment)."""
    vals = [float(r["inference_s"]) for r in rows if isinstance(r.get("inference_s"), (int, float))]
    if not vals:
        return None, None
    vals.sort()
    mean = statistics.mean(vals)
    n = len(vals)
    idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    return mean, vals[idx]
