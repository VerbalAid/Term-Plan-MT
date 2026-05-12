#!/usr/bin/env python3
r"""Paired segment bootstrap confidence intervals for Δ corpus BLEU (two results dirs).

Resamples segment indices with replacement, recomputes corpus BLEU on each resampled
slice for baseline vs. finetuned hypotheses against the same references, and forms
``Δ = BLEU_ft − BLEU_base`` per draw. Reports the point estimate and a 95% interval
(2.5/97.5 percentiles of the bootstrap distribution).

Requires aligned segment ids in both JSONL trees (same ``--segments`` + exclusion policy
as ``evaluate.py``).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import corpus_bleu
from metrics import fluency_hypothesis_text, load_results_jsonl
from metrics import eval_files_for_set
from systems import load_all_segments, parse_exclude_segment_ids


def _hyp_ref_by_id(
    results_dir: Path,
    fname: str,
    id_to_ref: dict[str, str],
    keep: frozenset[str],
) -> dict[str, str]:
    path = results_dir / fname
    if not path.is_file():
        raise FileNotFoundError(str(path))
    res = load_results_jsonl(path)
    out: dict[str, str] = {}
    for r in res:
        rid = str(r.get("id", ""))
        if rid in keep and rid in id_to_ref:
            out[rid] = fluency_hypothesis_text(r.get("hyp", ""))
    return out


def bootstrap_paired_delta_bleu(
    hyps_base: list[str],
    hyps_ft: list[str],
    refs: list[str],
    *,
    n_samples: int,
    seed: int,
) -> tuple[float, float, float]:
    assert len(hyps_base) == len(hyps_ft) == len(refs)
    n = len(refs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = corpus_bleu(hyps_ft, refs) - corpus_bleu(hyps_base, refs)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        hb = [hyps_base[i] for i in idx]
        hf = [hyps_ft[i] for i in idx]
        rr = [refs[i] for i in idx]
        deltas.append(corpus_bleu(hf, rr) - corpus_bleu(hb, rr))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1] if len(deltas) > 1 else deltas[0]
    return point, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--finetuned-dir", type=Path, required=True)
    ap.add_argument(
        "--segments",
        type=Path,
        required=True,
        help="Segment JSONL providing id → en_ref (e.g. data/section48/segments_ner_biollm.jsonl).",
    )
    ap.add_argument(
        "--exclude-segment-ids",
        type=str,
        default="",
        help="Comma-separated ids to drop (must match the scored runs). Use empty string for all segments.",
    )
    ap.add_argument(
        "--baseline-eval-file-set",
        choices=["standard", "mistral_clean"],
        default="standard",
    )
    ap.add_argument(
        "--finetuned-eval-file-set",
        choices=["standard", "mistral_clean"],
        default="mistral_clean",
    )
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV path (e.g. results/ner_biollm_finetuned/figures/bleu_delta_bootstrap_95ci.csv).",
    )
    args = ap.parse_args()

    seg_path = args.segments if args.segments.is_absolute() else (ROOT / args.segments)
    if not seg_path.is_file():
        raise SystemExit(f"Segments file not found: {seg_path}")
    exclude = parse_exclude_segment_ids(args.exclude_segment_ids or None)
    segment_rows = load_all_segments(seg_path, exclude_segment_ids=exclude)
    keep = frozenset(str(r["id"]) for r in segment_rows)
    id_to_ref = {str(r["id"]): r["en_ref"] for r in segment_rows}

    base_dir = args.baseline_dir if args.baseline_dir.is_absolute() else (ROOT / args.baseline_dir)
    ft_dir = args.finetuned_dir if args.finetuned_dir.is_absolute() else (ROOT / args.finetuned_dir)

    base_files = eval_files_for_set(args.baseline_eval_file_set)
    ft_by_label = {lab: fn for lab, fn in eval_files_for_set(args.finetuned_eval_file_set)}

    rows_out: list[dict[str, float | int | str]] = []
    for label, base_fn in base_files:
        ft_fn = ft_by_label.get(label)
        if ft_fn is None:
            print(f"[skip] no finetuned file mapping for {label}", file=sys.stderr)
            continue
        try:
            hb_map = _hyp_ref_by_id(base_dir, base_fn, id_to_ref, keep)
            hf_map = _hyp_ref_by_id(ft_dir, ft_fn, id_to_ref, keep)
        except FileNotFoundError as e:
            print(f"[skip] {label}: {e}", file=sys.stderr)
            continue
        common = sorted(set(hb_map) & set(hf_map))
        if not common:
            print(f"[skip] {label}: no overlapping segment ids", file=sys.stderr)
            continue
        if len(common) < len(keep):
            print(
                f"[skip] {label}: incomplete baseline or finetuned coverage "
                f"(only {len(common)}/{len(keep)} ids overlap; wait for full pipeline JSONL).",
                file=sys.stderr,
            )
            continue
        hb = [hb_map[i] for i in common]
        hf = [hf_map[i] for i in common]
        refs_b = [id_to_ref[i] for i in common]
        point, lo, hi = bootstrap_paired_delta_bleu(
            hb,
            hf,
            refs_b,
            n_samples=args.n_bootstrap,
            seed=args.seed,
        )
        rows_out.append(
            {
                "label": label,
                "n_segments": len(hb),
                "delta_bleu": round(point, 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
            }
        )
        print(
            f"{label:<12}  ΔBLEU={point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (n={len(hb)} segs, "
            f"B={args.n_bootstrap})"
        )

    if args.out_csv:
        if not rows_out:
            print(
                "No bootstrap rows computed (missing baseline JSONLs or no overlapping ids). "
                "CSV not written.",
                file=sys.stderr,
            )
        else:
            outp = args.out_csv if args.out_csv.is_absolute() else (ROOT / args.out_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["label", "n_segments", "delta_bleu", "ci95_low", "ci95_high"],
                )
                w.writeheader()
                w.writerows(rows_out)
            print(f"Wrote {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()