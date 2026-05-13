"""Standalone COMET scorer — run as a proper .py file to avoid Lightning stdin issues."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CONDITIONS = [
    {
        "results_dir": ROOT / "results" / "ner_biollm",
        "segments_file": ROOT / "data" / "section48" / "segments_ner_biollm.jsonl",
    },
    {
        "results_dir": ROOT / "results" / "ner_biollm_finetuned",
        "segments_file": ROOT / "data" / "section48" / "segments_ner_unsloth_full.jsonl",
    },
]

SYSTEMS = ["s1", "s2", "s3", "s4", "s5", "s5_mistral", "s6"]
EXCLUDE_IDS = {"48_028"}


def load_segments(path):
    segs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("segment_id") or obj.get("id")
            src = obj.get("fr") or obj.get("src") or obj.get("source") or ""
            ref = (
                obj.get("en_ref")
                or obj.get("ref")
                or obj.get("reference")
                or obj.get("en")
                or ""
            )
            if sid and sid not in EXCLUDE_IDS:
                segs[sid] = {"src": src, "ref": ref}
    return segs


def load_hyps(jsonl_path, segments):
    hyps = {}
    if not jsonl_path.exists():
        return hyps
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("segment_id") or obj.get("id")
            if sid and sid not in EXCLUDE_IDS and sid in segments:
                hyp = obj.get("hypothesis") or obj.get("hyp") or obj.get("translation") or ""
                # Strip contamination placeholder
                if "[TRANSLATION_PLACEHOLDER]" in hyp or "[SYSTEM_PROMPT]" in hyp:
                    hyp = ""
                if hyp:
                    hyps[sid] = hyp
    return hyps


def score_system(model, segments, hyps):
    common = sorted(set(segments) & set(hyps))
    if not common:
        return None
    data = [
        {"src": segments[sid]["src"], "mt": hyps[sid], "ref": segments[sid]["ref"]}
        for sid in common
    ]
    result = model.predict(data, batch_size=8, num_workers=0)
    return float(result.system_score)


def main():
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print(json.dumps({"error": "comet not installed"}))
        sys.exit(1)

    print("Loading COMET model...", file=sys.stderr)
    checkpoint = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(checkpoint)
    model.eval()
    print("COMET model loaded.", file=sys.stderr)

    output = {}

    for cond in CONDITIONS:
        rdir = cond["results_dir"]
        seg_file = cond["segments_file"]
        cond_key = rdir.name

        if not seg_file.exists():
            print(f"Segments not found: {seg_file}", file=sys.stderr)
            continue

        segments = load_segments(seg_file)
        print(f"\nCondition {cond_key}: {len(segments)} segments", file=sys.stderr)
        output[cond_key] = {}

        for sys_label in SYSTEMS:
            jsonl_path = rdir / f"{sys_label}.jsonl"
            hyps = load_hyps(jsonl_path, segments)
            if not hyps:
                print(f"  {sys_label}: no hypotheses found", file=sys.stderr)
                output[cond_key][sys_label] = None
                continue
            try:
                score = score_system(model, segments, hyps)
                print(f"  {sys_label}: {score:.4f} ({len(hyps)} segs)", file=sys.stderr)
                output[cond_key][sys_label] = score
            except Exception as e:
                print(f"  {sys_label}: ERROR {e}", file=sys.stderr)
                output[cond_key][sys_label] = None

    print(json.dumps(output))


if __name__ == "__main__":
    main()
