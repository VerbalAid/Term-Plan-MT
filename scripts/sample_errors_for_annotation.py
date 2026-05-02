#!/usr/bin/env python3
"""Sample MT outputs for manual error review (`docs/error_analysis/schema.md`).

Ranks `results/ner_*/*/s*.jsonl` by worst sentence chrF vs reference; default one row per segment.
Optional Ollama/OpenAI fills CSV columns using built-in MedDRA-aligned flag definitions.

Examples::

    PYTHONPATH=. python scripts/sample_errors_for_annotation.py \\
      --out-csv error_analysis/error_review_50.csv --n 50 --annotate-backend ollama

    PYTHONPATH=. python scripts/sample_errors_for_annotation.py \\
      --out-csv error_analysis/errors.csv --annotate-backend none
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.metrics.eval_manifest import EVAL_FILES

try:
    from sacrebleu.metrics import CHRF
except ImportError as e:  # pragma: no cover
    raise SystemExit("sacrebleu is required (pip install sacrebleu)") from e

_CHRF = CHRF()


class OllamaRequestError(RuntimeError):
    """Ollama /api/chat failed (HTTP, network, or unexpected payload)."""


ISSUE_CATEGORIES = [
    "terminology_wrong",
    "fluency",
    "omission",
    "addition",
    "ner_propagation",
    "other",
]

CSV_COLUMNS = [
    "segment_id",
    "ner_condition",
    "system_id",
    "issue_category",
    "severity",
    "source_span_fr",
    "hypothesis_span_en",
    "ref_span_en",
    "wrong_terminology",
    "concept_flattening_too_vague",
    "missing_terms",
    "unnatural_phrasing",
    "notes",
    "reviewer",
    "resolved",
]

# Shown to the annotator model so binary columns align with MedDRA-style ontology reasoning.
ERROR_REVIEW_ONTOLOGY_GUIDE = """
ONTOLOGY CONTEXT (MedDRA-aligned terminology for this study)

This project ties SmPC wording to MedDRA-like controlled concepts (tiers such as PT vs LLT and a
numeric level field appear in graph-backed hints). For error review, treat the reference English as
the regulatory gold: does the hypothesis preserve the same clinical meaning and specificity?

Use the four binary flags below only when the definition matches; multiple flags may be 1.

1) wrong_terminology — Wrong concept / wrong preferred rendering
   The hypothesis uses English that corresponds to a different medico-regulatory concept than the
   reference (different disorder, drug class, or mechanism/type of wording tied to a different branch of
   the terminology). This is substitution, not mere vagueness.

2) concept_flattening_too_vague — Specificity loss (flattening)
   The hypothesis uses broader or generic English than the reference while staying in a related
   neighbourhood of meaning: it drops a distinction the reference keeps (under-specific PT/LLT
   nuance, collapsed immune-mediated vs plain wording where the reference is narrower, loss of
   graded detail). Not a totally wrong branch — under-specific vs gold.

3) missing_terms — Omission of meaningful reference content
   Clinically meaningful words or phrases present in the reference are absent or only weakly
   implied in the hypothesis (content drop), beyond harmless reordering.

4) unnatural_phrasing — Non-native or awkward English
   Grammar, idioms, or SmPC register are off even if concepts largely align. Prefer this when the
   defect is fluency, not ontology mapping.

In notes, cite short quoted spans and say which numbered item (1–4) justified each flag set to 1.
""".strip()


def _flag01(v: Any) -> str:
    """Normalize LLM / CSV values to '0' or '1'."""
    if v is True:
        return "1"
    if v is False or v is None:
        return "0"
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return "1"
    return "0"


def _sentence_chrf(hyp: str, ref: str) -> float:
    hyp = (hyp or "").strip()
    ref = (ref or "").strip()
    if not hyp and not ref:
        return 100.0
    return float(_CHRF.sentence_score(hyp, [ref]).score)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _discover_rows(
    results_root: Path,
    ner_globs: list[str],
) -> list[tuple[float, dict[str, Any]]]:
    """Return list of (sentence_chrf, meta) for ranking (lower chrF = worse)."""
    scored: list[tuple[float, dict[str, Any]]] = []
    systems = [fn for _, fn in EVAL_FILES]
    for pattern in ner_globs:
        for ner_dir in sorted(results_root.glob(pattern)):
            if not ner_dir.is_dir() or not ner_dir.name.startswith("ner_"):
                continue
            cond = ner_dir.name
            for _sys_label, fname in EVAL_FILES:
                path = ner_dir / fname
                if not path.is_file():
                    continue
                for row in _load_jsonl(path):
                    hyp = row.get("hyp") or ""
                    ref = row.get("en_ref") or ""
                    sid = row.get("id") or ""
                    sys_id = row.get("system") or _sys_label
                    if not sid or not ref.strip():
                        continue
                    chrf = _sentence_chrf(hyp, ref)
                    scored.append(
                        (
                            chrf,
                            {
                                "segment_id": sid,
                                "ner_condition": cond,
                                "system_id": sys_id,
                                "fr": row.get("fr") or "",
                                "hyp": hyp,
                                "en_ref": ref,
                                "sentence_chrf": chrf,
                            },
                        )
                    )
    return scored


def _pick_worst(
    scored: list[tuple[float, dict[str, Any]]],
    *,
    n: int,
    max_per_segment: int,
    seed: int | None,
    unique_segment: bool,
) -> list[dict[str, Any]]:
    """Lowest chrF first. If ``unique_segment``, at most one row per ``segment_id``."""
    rng = random.Random(seed)
    enriched: list[tuple[float, dict[str, Any]]] = []
    for chrf, meta in scored:
        m = dict(meta)
        m["_tie"] = rng.random()
        enriched.append((chrf, m))
    enriched.sort(key=lambda x: (x[0], x[1]["_tie"]))

    counts: dict[str, int] = defaultdict(int)
    seen_seg: set[str] = set()
    picked: list[dict[str, Any]] = []
    for _chrf, meta in enriched:
        sid = meta["segment_id"]
        if unique_segment:
            if sid in seen_seg:
                continue
            seen_seg.add(sid)
        else:
            if counts[sid] >= max_per_segment:
                continue
            counts[sid] += 1
        meta.pop("_tie", None)
        picked.append(meta)
        if len(picked) >= n:
            break
    return picked


def _ollama_get_installed_models(*, base_url: str, timeout: float) -> list[str]:
    """Return model names from GET /api/tags (empty if unreachable)."""
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return []
    models = payload.get("models") or []
    out: list[str] = []
    for m in models:
        name = m.get("name") if isinstance(m, dict) else None
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def _ollama_model_is_present(installed: list[str], model: str) -> bool:
    if model in installed:
        return True
    m0 = model.split(":")[0]
    for n in installed:
        if n.startswith(model + ":") or n == m0:
            return True
        n0 = n.split(":")[0]
        if n0 == m0 or n0.endswith("/" + m0):
            return True
    return False


def _ollama_preflight_message(installed: list[str], model: str, base_url: str) -> str:
    sample = ", ".join(installed[:16]) + (" …" if len(installed) > 16 else "")
    replace_hint = (
        f"  Or pass an installed name, e.g. --ollama-model {installed[0]!r}\n"
        if installed
        else f"  Run: ollama pull {model}   # only if that tag exists in the library\n"
    )
    return (
        f"Model {model!r} was not found among local Ollama tags at {base_url.rstrip('/')}/api/tags.\n"
        f"  Installed (sample): {sample or '(none)'}\n"
        "  Run: ollama serve   # if needed\n"
        "  Run: ollama pull <exact-name>   # see https://ollama.com/library ; names must match `ollama list`\n"
        f"{replace_hint}"
        "  Or use --annotate-backend none to skip the LLM."
    )


def _ollama_chat(
    *,
    base_url: str,
    model: str,
    user_content: str,
    system: str | None,
    timeout: float | None,
    num_predict: int,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    messages: list[dict[str, str]] = []
    sys_t = (system or "").strip()
    if sys_t:
        messages.append({"role": "system", "content": sys_t})
    messages.append({"role": "user", "content": user_content})
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max(128, int(num_predict)), "temperature": 0},
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise OllamaRequestError(
                f"HTTP 404 — no model {model!r} on Ollama (pull the exact tag or fix --ollama-model). "
                f"Try: ollama list"
            ) from e
        raise OllamaRequestError(f"Ollama HTTP {e.code}: {e}") from e
    except urllib.error.URLError as e:
        raise OllamaRequestError(f"Ollama request failed (is ollama serve running?): {e}") from e
    msg = payload.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise OllamaRequestError(f"Unexpected Ollama response: {payload!r}")
    return content


def _openai_chat(*, api_key: str, model: str, user_content: str, timeout: float) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"OpenAI HTTP {e.code}: {e.read().decode(errors='replace')}") from e
    choices = payload.get("choices") or []
    if not choices:
        raise SystemExit(f"OpenAI empty choices: {payload!r}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise SystemExit(f"Unexpected OpenAI response: {payload!r}")
    return content


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


def _annotate_row_llm(
    meta: dict[str, Any],
    *,
    backend: str,
    ollama_base: str,
    ollama_model: str,
    ollama_system: str,
    openai_model: str,
    timeout: float | None,
    ontology_guide: str,
) -> dict[str, str]:
    cats = ", ".join(ISSUE_CATEGORIES)
    guide_block = (ontology_guide.strip() + "\n\n") if ontology_guide.strip() else ""
    prompt = f"""{guide_block}You annotate French→English medical SmPC MT errors for a qualitative study.

French source (segment excerpt possible):
---
{meta["fr"][:8000]}
---

English reference:
---
{meta["en_ref"][:8000]}
---

System hypothesis ({meta["system_id"]}):
---
{meta["hyp"][:8000]}
---

Sentence chrF vs reference (lower = worse overlap): {meta["sentence_chrf"]:.2f}

Apply the ontology guide definitions when setting the four binary flags below.

Reply with a **single JSON object only** (no markdown fences), keys:
- "issue_category": one of [{cats}]
- "severity": one of "minor", "major", "critical" or null
- "source_span_fr": short French substring illustrating the issue, or ""
- "hypothesis_span_en": short English substring from the hypothesis, or ""
- "ref_span_en": short English substring from the reference for contrast, or ""
- "wrong_terminology": 0 or 1 — item 1 in the ontology guide
- "concept_flattening_too_vague": 0 or 1 — item 2 (specificity loss)
- "missing_terms": 0 or 1 — item 3
- "unnatural_phrasing": 0 or 1 — item 4
- "notes": one or two sentences citing spans and which items (1–4) were applied.

JSON:"""

    if backend == "ollama":
        raw = _ollama_chat(
            base_url=ollama_base,
            model=ollama_model,
            user_content=prompt,
            system=ollama_system or None,
            timeout=timeout,
            num_predict=640,
        )
    elif backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set.")
        raw = _openai_chat(api_key=key, model=openai_model, user_content=prompt, timeout=timeout or 120.0)
    else:
        raise SystemExit(f"Unknown backend {backend}")

    try:
        obj = _extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logging.warning("LLM JSON parse failed; using fallback notes: %s", e)
        obj = {
            "issue_category": "other",
            "severity": None,
            "source_span_fr": "",
            "hypothesis_span_en": "",
            "ref_span_en": "",
            "wrong_terminology": 0,
            "concept_flattening_too_vague": 0,
            "missing_terms": 0,
            "unnatural_phrasing": 0,
            "notes": f"[parse_error] Raw model output (trimmed): {raw[:500]}",
        }

    ic = str(obj.get("issue_category") or "other").strip()
    if ic not in ISSUE_CATEGORIES:
        ic = "other"
    sev = obj.get("severity")
    sev_s = "" if sev is None else str(sev).strip()
    if sev_s not in ("minor", "major", "critical", ""):
        sev_s = ""

    return {
        "issue_category": ic,
        "severity": sev_s,
        "source_span_fr": str(obj.get("source_span_fr") or "")[:2000],
        "hypothesis_span_en": str(obj.get("hypothesis_span_en") or "")[:2000],
        "ref_span_en": str(obj.get("ref_span_en") or "")[:2000],
        "wrong_terminology": _flag01(obj.get("wrong_terminology")),
        "concept_flattening_too_vague": _flag01(obj.get("concept_flattening_too_vague")),
        "missing_terms": _flag01(obj.get("missing_terms")),
        "unnatural_phrasing": _flag01(obj.get("unnatural_phrasing")),
        "notes": str(obj.get("notes") or "")[:4000],
    }


def _annotate_heuristic(meta: dict[str, Any]) -> dict[str, str]:
    ch = meta["sentence_chrf"]
    return {
        "issue_category": "fluency",
        "severity": "major" if ch < 15 else "minor",
        "source_span_fr": "",
        "hypothesis_span_en": "",
        "ref_span_en": "",
        "wrong_terminology": "0",
        "concept_flattening_too_vague": "0",
        "missing_terms": "0",
        "unnatural_phrasing": "0",
        "notes": (
            f"Heuristic sample: sentence chrF vs reference = {ch:.2f} "
            f"(condition={meta['ner_condition']}, system={meta['system_id']}). Replace with human judgement."
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Sample worst-overlap MT rows and export annotation CSV (schema-aligned).",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="Directory containing ner_* folders (default: results/).",
    )
    ap.add_argument(
        "--ner-glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob under results-root (repeatable). Default: ner_*",
    )
    ap.add_argument("--n", type=int, default=50, help="Number of rows to export (default 50).")
    ap.add_argument(
        "--max-per-segment",
        type=int,
        default=3,
        help="When --repeat-segments: max rows per segment_id for diversity (default 3).",
    )
    ap.add_argument(
        "--repeat-segments",
        action="store_true",
        help=(
            "Allow multiple CSV rows per segment_id (different systems/conditions). "
            "Default: one row per segment_id so --n is ‘50 segments’."
        ),
    )
    ap.add_argument(
        "--omit-ontology-guide",
        action="store_true",
        help="Do not prepend MedDRA / specificity definitions to LLM prompts (not recommended).",
    )
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for tie-breaking.")
    ap.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path (parent dirs created).",
    )
    ap.add_argument(
        "--annotate-backend",
        choices=("none", "ollama", "openai"),
        default="none",
        help="Pre-fill annotation fields: none | local Ollama | OpenAI API.",
    )
    ap.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "llama3.2"))
    ap.add_argument(
        "--ollama-system",
        default=None,
        metavar="TEXT",
        help=(
            "Optional system prompt for Ollama /api/chat (role=system). "
            "If omitted, uses OLLAMA_SYSTEM_PROMPT env when set. "
            "Does not change weights; stacks with fine-tuned behaviour."
        ),
    )
    ap.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    ap.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout seconds for LLM calls (default 180).",
    )
    ap.add_argument(
        "--ollama-strict-preflight",
        action="store_true",
        help="Exit before sampling if GET /api/tags shows --ollama-model is not installed.",
    )
    ap.add_argument(
        "--no-ollama-fallback-on-error",
        dest="ollama_fallback_on_error",
        action="store_false",
        default=True,
        help="Abort on Ollama failure instead of filling that row with heuristic annotations.",
    )
    args = ap.parse_args()

    ollama_system = (
        args.ollama_system.strip()
        if args.ollama_system is not None
        else os.environ.get("OLLAMA_SYSTEM_PROMPT", "").strip()
    )

    results_root = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    globs = args.ner_glob if args.ner_glob else ["ner_*"]

    scored = _discover_rows(results_root, globs)
    if not scored:
        raise SystemExit(f"No pipeline JSONLs found under {results_root}/*/s*.jsonl")

    picked = _pick_worst(
        scored,
        n=args.n,
        max_per_segment=args.max_per_segment,
        seed=args.seed,
        unique_segment=not args.repeat_segments,
    )
    logging.info("Selected %d rows from %d scored triples.", len(picked), len(scored))

    if args.annotate_backend == "ollama":
        installed = _ollama_get_installed_models(base_url=args.ollama_base_url, timeout=min(15.0, args.timeout))
        if installed and not _ollama_model_is_present(installed, args.ollama_model):
            msg = _ollama_preflight_message(installed, args.ollama_model, args.ollama_base_url)
            if args.ollama_strict_preflight:
                raise SystemExit(msg)
            logging.warning("%s\nContinuing (--ollama-strict-preflight not set); rows may use heuristic fallback.", msg)
        elif not installed:
            logging.warning(
                "Could not reach Ollama at %s/api/tags (empty list). Is `ollama serve` running? "
                "Annotating may fail per row; heuristic fallback applies if enabled.",
                args.ollama_base_url.rstrip("/"),
            )

    ontology_guide = "" if args.omit_ontology_guide else ERROR_REVIEW_ONTOLOGY_GUIDE

    rows_out: list[dict[str, str]] = []
    for i, meta in enumerate(picked):
        if args.annotate_backend == "none":
            ann = _annotate_heuristic(meta)
            reviewer = "heuristic"
        elif args.annotate_backend == "ollama":
            logging.info("Annotating %d/%d via Ollama…", i + 1, len(picked))
            try:
                ann = _annotate_row_llm(
                    meta,
                    backend="ollama",
                    ollama_base=args.ollama_base_url,
                    ollama_model=args.ollama_model,
                    ollama_system=ollama_system,
                    openai_model=args.openai_model,
                    timeout=args.timeout,
                    ontology_guide=ontology_guide,
                )
                reviewer = f"auto:ollama:{args.ollama_model}"
            except OllamaRequestError as e:
                if not args.ollama_fallback_on_error:
                    raise SystemExit(str(e)) from e
                logging.warning("Ollama failed for row %d/%d: %s — using heuristic fallback.", i + 1, len(picked), e)
                ann = _annotate_heuristic(meta)
                ann["notes"] = f"[ollama_failed] {e}\n" + ann["notes"]
                reviewer = "heuristic_fallback"
        else:
            logging.info("Annotating %d/%d via OpenAI…", i + 1, len(picked))
            ann = _annotate_row_llm(
                meta,
                backend="openai",
                ollama_base=args.ollama_base_url,
                ollama_model=args.ollama_model,
                ollama_system="",
                openai_model=args.openai_model,
                timeout=args.timeout,
                ontology_guide=ontology_guide,
            )
            reviewer = f"auto:openai:{args.openai_model}"

        rows_out.append(
            {
                "segment_id": meta["segment_id"],
                "ner_condition": meta["ner_condition"],
                "system_id": meta["system_id"],
                "issue_category": ann["issue_category"],
                "severity": ann["severity"],
                "source_span_fr": ann["source_span_fr"],
                "hypothesis_span_en": ann["hypothesis_span_en"],
                "ref_span_en": ann["ref_span_en"],
                "wrong_terminology": ann["wrong_terminology"],
                "concept_flattening_too_vague": ann["concept_flattening_too_vague"],
                "missing_terms": ann["missing_terms"],
                "unnatural_phrasing": ann["unnatural_phrasing"],
                "notes": ann["notes"],
                "reviewer": reviewer,
                "resolved": "0",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    og_note = ""
    if args.annotate_backend != "none":
        og_note = f" Ontology guide in LLM prompts: {'no' if args.omit_ontology_guide else 'yes'}."
    print(
        f"Wrote {args.out_csv} ({len(rows_out)} rows). "
        "Schema: docs/error_analysis/schema.md (+ ner_condition)."
        f"{og_note}"
    )


if __name__ == "__main__":
    main()
