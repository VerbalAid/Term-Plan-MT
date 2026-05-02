#!/usr/bin/env python3
"""Extract MedDRA English/French ASCII zips into data/meddra/ (requires MEDDRA_ZIP_PASSWORD in .env)."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _extract_filtered(zf: zipfile.ZipFile, meddra: Path, prefix: str) -> int:
    n = 0
    for name in zf.namelist():
        if not name.startswith(prefix) or not name.endswith(".asc"):
            continue
        zf.extract(name, meddra)
        n += 1
    return n


def main() -> None:
    pwd = os.environ.get("MEDDRA_ZIP_PASSWORD")
    if not pwd:
        print(
            "Set MEDDRA_ZIP_PASSWORD in .env to the password supplied with the MedDRA subscription, "
            "then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    meddra = ROOT / "data" / "meddra"
    meddra.mkdir(parents=True, exist_ok=True)
    pwd_b = pwd.encode("utf-8")

    en_zips = sorted(meddra.glob("MedDRA_*_English.zip"))
    fr_zips = sorted(meddra.glob("MedDRA_*_French.zip"))
    if not en_zips or not fr_zips:
        print("Expected zips: data/meddra/MedDRA_*_English.zip and MedDRA_*_French.zip", file=sys.stderr)
        sys.exit(1)

    en_zip, fr_zip = en_zips[0], fr_zips[0]
    try:
        with zipfile.ZipFile(en_zip) as zf:
            zf.setpassword(pwd_b)
            n_en = _extract_filtered(zf, meddra, "MedAscii/")
        with zipfile.ZipFile(fr_zip) as zf:
            zf.setpassword(pwd_b)
            n_fr = _extract_filtered(zf, meddra, "ascii-290/")
    except RuntimeError as e:
        if "Bad password" in str(e) or "incorrect" in str(e).lower():
            print(
                "Zip password rejected. Use the exact per-file password from the MedDRA 29.0 "
                "!!readme in the download (MSSO) — it is not always the same as the subscriber ID or CRID.",
                file=sys.stderr,
            )
        raise

    print(f"Extracted {n_en} English .asc files → {meddra / 'MedAscii'}")
    print(f"Extracted {n_fr} French .asc files → {meddra / 'ascii-290'}")


if __name__ == "__main__":
    main()
