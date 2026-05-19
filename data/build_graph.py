#!/usr/bin/env python3
"""Load MedDRA flat files into Neo4j when available; optional French overlay from a JSON seed list (`data/gold_terms.json`)."""


from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from pipeline import (
    enrich_mdhier_row_pt,
    load_llt_to_parent_pt,
    load_pt_names,
    parse_mdhier_row,
    read_meddra_asc,
    split_meddra_asc_line,
)


def ensure_concept_indexes(session) -> None:
    """Speed up MERGE/MATCH on :Concept(id) and fr_label updates (avoids full scans on ~120k nodes)."""
    session.run("CREATE RANGE INDEX concept_id IF NOT EXISTS FOR (c:Concept) ON (c.id)")
    session.run("CREATE RANGE INDEX concept_fr_label IF NOT EXISTS FOR (c:Concept) ON (c.fr_label)")


def load_level_file(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in read_meddra_asc(path).splitlines():
        if not line.strip():
            continue
        parts = split_meddra_asc_line(line)
        if len(parts) < 2:
            continue
        rows.append((parts[0], parts[1]))
    return rows


def load_code_pair_file(path: Path) -> list[tuple[str, str]]:
    """MedDRA relation files: ``parent_code$child_code$`` (e.g. hlt_pt, soc_hlgt)."""
    rows: list[tuple[str, str]] = []
    for line in read_meddra_asc(path).splitlines():
        if not line.strip():
            continue
        parts = split_meddra_asc_line(line)
        if len(parts) >= 2:
            parent, child = parts[0].strip(), parts[1].strip()
            if parent.isdigit() and child.isdigit():
                rows.append((parent, child))
    return rows


def _add_edge(edges: set[tuple[str, str]], parent: str, child: str) -> None:
    if parent and child and parent != child:
        edges.add((parent, child))


_TIER_RANK = {"SOC": 1, "HLGT": 2, "HLT": 3, "PT": 4, "LLT": 5}


def _merge_concept(
    concepts: dict[str, tuple[str, int, str]],
    cid: str,
    name: str,
    level: int,
    tier: str,
) -> None:
    """Keep the broadest tier when the same MedDRA code appears in multiple level files."""
    if cid in concepts:
        _name, old_level, old_tier = concepts[cid]
        old_rank = _TIER_RANK.get(old_tier, old_level)
        new_rank = _TIER_RANK.get(tier, level)
        if new_rank >= old_rank:
            return
    concepts[cid] = (name, level, tier)


def resolve_english_asc_dir(meddra: Path) -> Path | None:
    nested = meddra / "MedAscii" / "mdhier.asc"
    if nested.is_file():
        return meddra / "MedAscii"
    flat = meddra / "mdhier.asc"
    if flat.is_file():
        return meddra
    return None


def resolve_french_asc_dir(meddra: Path) -> Path | None:
    d = meddra / "ascii-290"
    if (d / "pt.asc").is_file() and (d / "llt.asc").is_file():
        return d
    return None


def extract_meddra_zips_if_configured(meddra: Path) -> None:
    """If zips are present but ASCII folders are missing, extract using MEDDRA_ZIP_PASSWORD."""
    pwd = os.environ.get("MEDDRA_ZIP_PASSWORD")
    if resolve_english_asc_dir(meddra) and resolve_french_asc_dir(meddra):
        return
    if not pwd:
        if list(meddra.glob("MedDRA_*_English.zip")) or list(meddra.glob("MedDRA_*_French.zip")):
            print(
                "MedDRA .zip files found but ASCII not extracted. "
                "Set MEDDRA_ZIP_PASSWORD in .env, then run: python data/extract_meddra.py"
            )
        return

    pwd_b = pwd.encode("utf-8")

    def pull(zpattern: str, inner_prefix: str) -> None:
        zips = sorted(meddra.glob(zpattern))
        if not zips:
            return
        with zipfile.ZipFile(zips[0]) as zf:
            zf.setpassword(pwd_b)
            for name in zf.namelist():
                if name.startswith(inner_prefix) and name.endswith(".asc"):
                    zf.extract(name, meddra)

    pull("MedDRA_*_English.zip", "MedAscii/")
    pull("MedDRA_*_French.zip", "ascii-290/")


def apply_french_labels(session, fr_dir: Path, batch_size: int = 4000) -> None:
    """Set Concept.fr_label from French MedDRA level files (same MedDRA codes as English)."""
    for fname in ("soc.asc", "hlgt.asc", "hlt.asc", "pt.asc", "llt.asc"):
        path = fr_dir / fname
        if not path.is_file():
            continue
        batch: list[dict] = []
        for line in read_meddra_asc(path).splitlines():
            if not line.strip():
                continue
            parts = split_meddra_asc_line(line)
            if len(parts) < 2:
                continue
            batch.append({"id": parts[0], "fr": parts[1]})
            if len(batch) >= batch_size:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Concept {id: row.id})
                    SET c.fr_label = row.fr
                    """,
                    rows=batch,
                )
                batch.clear()
        if batch:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Concept {id: row.id})
                SET c.fr_label = row.fr
                """,
                rows=batch,
            )
    print("Applied French labels from", fr_dir)


def load_meddra_english(session, en_dir: Path) -> None:
    mdhier_path = en_dir / "mdhier.asc"
    print("Loading MedDRA (English) from", en_dir)
    concepts: dict[str, tuple[str, int, str]] = {}
    edges: set[tuple[str, str]] = set()

    llt_path = en_dir / "llt.asc"
    pt_path = en_dir / "pt.asc"
    llt_pt = load_llt_to_parent_pt(llt_path) if llt_path.is_file() else {}
    pt_names = load_pt_names(pt_path) if pt_path.is_file() else {}

    for line in read_meddra_asc(mdhier_path).splitlines():
        if not line.strip():
            continue
        row = parse_mdhier_row(split_meddra_asc_line(line))
        if not row:
            continue
        if (row.get("primary_soc_fg") or "Y").upper() != "Y":
            continue
        row = enrich_mdhier_row_pt(row, llt_pt, pt_names)
        pt_code = (row.get("pt_code") or "").strip()
        if not pt_code:
            continue
        _merge_concept(concepts, row["soc_code"], row["soc_name"], 1, "SOC")
        _merge_concept(concepts, row["hlgt_code"], row["hlgt_name"], 2, "HLGT")
        _merge_concept(concepts, row["hlt_code"], row["hlt_name"], 3, "HLT")
        _merge_concept(concepts, pt_code, row["pt_name"], 4, "PT")
        _merge_concept(concepts, row["llt_code"], row["llt_name"], 5, "LLT")

        _add_edge(edges, row["soc_code"], row["hlgt_code"])
        _add_edge(edges, row["hlgt_code"], row["hlt_code"])
        _add_edge(edges, row["hlt_code"], pt_code)
        _add_edge(edges, pt_code, row["llt_code"])

    for rel_file in ("soc_hlgt.asc", "hlgt_hlt.asc", "hlt_pt.asc"):
        rel_path = en_dir / rel_file
        if rel_path.is_file():
            for parent, child in load_code_pair_file(rel_path):
                _add_edge(edges, parent, child)

    for llt_code, pt_code in llt_pt.items():
        _add_edge(edges, pt_code, llt_code)

    for fname, level, tier in [
        ("soc.asc", 1, "SOC"),
        ("hlgt.asc", 2, "HLGT"),
        ("hlt.asc", 3, "HLT"),
        ("pt.asc", 4, "PT"),
        ("llt.asc", 5, "LLT"),
    ]:
        p = en_dir / fname
        if not p.exists():
            continue
        for cid, cname in load_level_file(p):
            _merge_concept(concepts, cid, cname, level, tier)

    session.run(
        """
        UNWIND $rows AS row
        MERGE (c:Concept {id: row.id})
        SET c.name = row.name, c.level = row.level, c.tier = row.tier
        """,
        rows=[
            {"id": cid, "name": name, "level": lvl, "tier": tier}
            for cid, (name, lvl, tier) in concepts.items()
        ],
    )
    session.run(
        """
        UNWIND $rels AS rel
        MATCH (p:Concept {id: rel.parent}), (c:Concept {id: rel.child})
        MERGE (p)-[:BROADER_THAN]->(c)
        """,
        rels=[{"parent": a, "child": b} for a, b in sorted(edges)],
    )


def main() -> None:
    meddra = ROOT / "data" / "meddra"
    gold_path = ROOT / "data" / "gold_terms.json"
    gold: list[dict[str, Any]] = []
    if gold_path.is_file():
        gold = json.loads(gold_path.read_text(encoding="utf-8"))

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASS", "password")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
    except ServiceUnavailable:
        driver.close()
        print(
            "Neo4j is not reachable at "
            + uri
            + ". Start the database first, e.g. from this project directory:\n"
            "  docker compose up -d\n"
            "Then wait a few seconds and run this script again.",
            file=sys.stderr,
        )
        sys.exit(1)

    with driver.session() as session:
        ensure_concept_indexes(session)
        session.run("MATCH (n) DETACH DELETE n")

        loaded_meddra = False
        en_dir = resolve_english_asc_dir(meddra)
        if en_dir:
            load_meddra_english(session, en_dir)
            fr_dir = resolve_french_asc_dir(meddra)
            if fr_dir:
                apply_french_labels(session, fr_dir)
            else:
                print("No French ascii-290/ folder — fr_label will come from optional seed JSON only where matched.")

            for g in gold:
                session.run(
                    """
                    MATCH (c:Concept)
                    WHERE toLower(c.name) = toLower($en_label)
                    SET c.fr_label = $fr_label
                    """,
                    en_label=g["en_label"],
                    fr_label=g["fr"],
                )
            loaded_meddra = True
        else:
            print("No English mdhier.asc found under data/meddra/ — seeding from optional seed JSON only.")

        if not loaded_meddra:
            if not gold:
                driver.close()
                print(
                    "No MedDRA English tree under data/meddra/ and no optional seed JSON file — nothing to load.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("Seeding concepts from seed JSON list")
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Concept {name: row.en_label})
                SET c.fr_label = row.fr_label,
                    c.level = row.level,
                    c.tier = row.tier,
                    c.id = row.en_label
                """,
                rows=[
                    {
                        "en_label": g["en_label"],
                        "fr_label": g["fr"],
                        "level": g["level"],
                        "tier": g["tier"],
                    }
                    for g in gold
                ],
            )

    driver.close()
    print("Graph build complete.")


if __name__ == "__main__":
    main()
