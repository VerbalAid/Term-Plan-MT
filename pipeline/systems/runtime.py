"""Repo path bootstrap and shared TermGraph lifecycle for system runners."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_on_syspath() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


@contextmanager
def term_graph_session(graph: Any) -> Iterator[Any]:
    from pipeline.graph import TermGraph

    if graph is not None:
        yield graph
    else:
        g = TermGraph()
        try:
            yield g
        finally:
            g.close()
