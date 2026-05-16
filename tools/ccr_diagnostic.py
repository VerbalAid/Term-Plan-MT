"""
TermPlanMT — CCR (Concept Coverage Rate) Diagnostic Tool

Loads NER terms from a segments JSONL file, attempts to ground each unique
term against the MedDRA Neo4j graph (via TermGraph in pipeline.py), and
classifies each term as:

  EXACT_HIT    — normalised string matches a node in the graph exactly
  FUZZY_HIT    — matched via RapidFuzz fuzzy fallback (PT/LLT tier only)
  AMBIGUOUS    — grounded but the French label maps to multiple concepts
  NOT_IN_GRAPH — no match above the fuzzy cutoff

Usage:
  python3 -m tools.ccr_diagnostic --segments data/section48/segments_ner_biollm.jsonl \\
                                   --out results/ccr_diagnostic.txt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── Classification logic ──────────────────────────────────────────────────────

_CATEGORIES = ("EXACT_HIT", "FUZZY_HIT", "AMBIGUOUS", "NOT_IN_GRAPH")


def _classify_term(
    term: str,
    graph,            # TermGraph instance (or None if unavailable)
    norm_keys: set,   # set of normalised keys from graph cache (for exact check)
) -> str:
    """Return one of the four category strings for *term*."""
    if graph is None:
        return "NOT_IN_GRAPH"

    # Check ambiguity first — is_ambiguous() triggers _load() lazily.
    if graph.is_ambiguous(term):
        return "AMBIGUOUS"

    # Attempt grounding.
    hit = graph.ground(term)
    if hit is None:
        return "NOT_IN_GRAPH"

    # Distinguish exact vs fuzzy: exact if the normalised key is in the cache.
    from pipeline import _norm  # local import to keep module importable without Neo4j
    key = _norm(term)
    if key in norm_keys:
        return "EXACT_HIT"
    return "FUZZY_HIT"


def _get_norm_keys(graph) -> set:
    """Retrieve the set of normalised cache keys from a loaded TermGraph."""
    try:
        graph._load()  # ensure cache is populated
        return set(graph._cache.keys()) if graph._cache else set()
    except Exception:
        return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify NER terms from a segments JSONL against the MedDRA graph."
    )
    parser.add_argument("--segments", required=True,
                        help="Path to segments JSONL file (must have a 'terms' field per row).")
    parser.add_argument("--out", required=True,
                        help="Path to write the diagnostic report.")
    args = parser.parse_args()

    segs_path = Path(args.segments)
    if not segs_path.exists():
        print(f"ERROR: segments file not found: {segs_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load segments ─────────────────────────────────────────────────────────
    segments = [
        json.loads(line)
        for line in segs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(segments)} segments from {segs_path}")

    # Collect unique terms and their frequencies.
    term_freq: dict[str, int] = defaultdict(int)
    for seg in segments:
        for t in seg.get("terms") or []:
            word = (t.get("word") or "").strip()
            if word:
                term_freq[word] += 1

    unique_terms = sorted(term_freq.keys())
    print(f"Found {len(unique_terms)} unique NER terms across {sum(term_freq.values())} occurrences")

    # ── Attempt to connect to Neo4j via TermGraph ─────────────────────────────
    graph = None
    norm_keys: set = set()
    neo4j_available = False

    try:
        from pipeline import TermGraph
        graph = TermGraph(grounding_mode="string")
        norm_keys = _get_norm_keys(graph)
        neo4j_available = True
        print(f"Neo4j connected — {len(norm_keys)} concept keys loaded")
    except Exception as exc:
        # Neo4j not running or pipeline import error — degrade gracefully.
        print(
            f"\nWARNING: Could not connect to Neo4j ({type(exc).__name__}: {exc}).\n"
            "All terms will be classified as NOT_IN_GRAPH.\n",
            file=sys.stderr,
        )

    # ── Classify each term ────────────────────────────────────────────────────
    counts: dict[str, int] = {cat: 0 for cat in _CATEGORIES}
    rows: list[tuple[str, str, int]] = []  # (term, category, freq)

    for term in unique_terms:
        cat = _classify_term(term, graph, norm_keys)
        counts[cat] += 1
        rows.append((term, cat, term_freq[term]))

    total = len(unique_terms)

    # ── Print breakdown table ─────────────────────────────────────────────────
    sep = "=" * 62
    header_line = f"{'TERM':<35} {'CATEGORY':<16} {'FREQ':>5}"

    lines = [
        "CCR DIAGNOSTIC REPORT",
        f"Segments: {segs_path}",
        f"Unique terms: {total}  |  Neo4j: {'available' if neo4j_available else 'UNAVAILABLE'}",
        sep,
        "",
        "CATEGORY BREAKDOWN",
        "-" * 40,
    ]
    for cat in _CATEGORIES:
        pct = round(100 * counts[cat] / total) if total else 0
        lines.append(f"  {cat:<20} {counts[cat]:>4}/{total} = {pct:>3}%")

    lines += [
        "",
        sep,
        "TERM-LEVEL DETAIL",
        header_line,
        "-" * 62,
    ]
    for term, cat, freq in sorted(rows, key=lambda x: (_CATEGORIES.index(x[1]), x[0])):
        lines.append(f"{term:<35} {cat:<16} {freq:>5}")

    report = "\n".join(lines) + "\n"

    print()
    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
