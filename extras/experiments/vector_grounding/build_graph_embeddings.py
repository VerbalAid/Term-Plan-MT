#!/usr/bin/env python3
"""Embed MedDRA Concept.fr_label into Neo4j (vector property + optional vector index)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
VECTOR_DIM = 768
INDEX_NAME = "meddra_fr_embedding"


def main() -> None:
    p = argparse.ArgumentParser(description="Embed Concept.fr_label → c.fr_embedding in Neo4j.")
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("TERMPLAN_EMBED_MODEL", DEFAULT_MODEL),
        help="sentence-transformers model id (default: mpnet multilingual).",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N concepts (debug).",
    )
    p.add_argument(
        "--no-index",
        action="store_true",
        help="Skip CREATE VECTOR INDEX (embeddings only).",
    )
    args = p.parse_args()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASS", "password")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    model = SentenceTransformer(args.model)

    fetch_q = """
    MATCH (c:Concept)
    WHERE c.fr_label IS NOT NULL AND trim(toString(c.fr_label)) <> ''
    RETURN elementId(c) AS eid, c.id AS cid, c.fr_label AS fr_label
    ORDER BY elementId(c)
    """
    lim = int(args.limit) if args.limit is not None else None
    params: dict[str, Any] = {}
    if lim is not None:
        fetch_q += "\nLIMIT $lim"
        params["lim"] = lim

    with driver.session() as session:
        rows = list(session.run(fetch_q, **params))
        labels = [r["fr_label"] for r in rows]
        eids = [r["eid"] for r in rows]

    if not labels:
        print("No Concept nodes with fr_label found.", file=sys.stderr)
        driver.close()
        raise SystemExit(1)

    n_done = 0
    bs = max(1, args.batch_size)
    with driver.session() as session:
        for start in tqdm(range(0, len(labels), bs), desc="Embedding batches"):
            batch_labs = labels[start : start + bs]
            batch_eids = eids[start : start + bs]
            emb = model.encode(batch_labs, batch_size=len(batch_labs), show_progress_bar=False)
            rows_param = [
                {"eid": eid, "emb": [float(x) for x in vec]}
                for eid, vec in zip(batch_eids, emb)
            ]
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Concept)
                WHERE elementId(c) = row.eid
                SET c.fr_embedding = row.emb
                """,
                rows=rows_param,
            )
            n_done += len(rows_param)

    if not args.no_index:
        idx_cypher = f"""
CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
FOR (c:Concept) ON (c.fr_embedding)
OPTIONS {{
 indexConfig: {{
  `vector.dimensions`: {VECTOR_DIM},
  `vector.similarity_function`: 'cosine'
 }}
}}
"""
        try:
            with driver.session() as session:
                session.run(idx_cypher)
            print(f"Vector index '{INDEX_NAME}' created or already exists.")
        except Exception as e:
            print(
                f"warning: could not create vector index (Neo4j 5.13+ required): {e}",
                file=sys.stderr,
            )

    driver.close()
    print(f"Embedded {n_done} Concept nodes with fr_embedding ({args.model}).")


if __name__ == "__main__":
    main()
