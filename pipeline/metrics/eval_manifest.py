"""Which systems map to which results files (single source for evaluate + plots)."""

from __future__ import annotations

EVAL_FILES: list[tuple[str, str]] = [
    ("s1", "s1.jsonl"),
    ("s2", "s2.jsonl"),
    ("s3", "s3.jsonl"),
    ("s4", "s4.jsonl"),
    ("s5", "s5.jsonl"),
    ("s5_mistral", "s5_mistral.jsonl"),
]

# rerun_all.sh / run_eval_plot_matrix: (results_subdir, segment_jsonl_candidates).
# NER spans for HTM/CCR come from the first existing JSONL in each tuple.
EVAL_RERUN_PROFILES: list[tuple[str, tuple[str, ...]]] = [
    ("results/ner_biollm", ("data/section48/segments_ner_biollm.jsonl",)),
    (
        "results/ner_biollm_finetuned",
        (
            "data/section48/segments_ner_unsloth.jsonl",
            "data/section48/segments_ner_unsloth_full.jsonl",
        ),
    ),
    # Historical / appendix NER stacks (same SmPC §4.8 segments; NER list from ``segments_ner.jsonl``).
    ("results/ner_baseline", ("data/section48/segments_ner.jsonl",)),
    ("results/ner_finetuned", ("data/section48/segments_ner.jsonl",)),
]


def condition_name_from_results_subdir(results_subdir: str) -> str:
    """``results/ner_biollm`` → ``ner_biollm`` (folder name under ``results/``)."""
    p = results_subdir.strip().strip("/")
    if p.startswith("results/"):
        return p[len("results/") :]
    return p
