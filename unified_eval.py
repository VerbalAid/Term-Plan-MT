"""
TermPlanMT Unified Expert Evaluation — thin wrapper around eval_pipeline.py

Evaluates the same 30 segments as audit_pipeline.py (seed=42, IDs from
annotations/audit_annotated.csv) across all 7 non-oracle systems, both NER conditions.
"""

import subprocess, sys

subprocess.run(
    [
        sys.executable, "eval_pipeline.py",
        "--segments",   "30",
        "--seed",       "42",
        "--use-ids",    "error_analysis/annotations/audit_annotated.csv",
        "--systems",    "s1,s2,s3,s3ft,s5b,s5ft,s6",
        "--out-csv",    "error_analysis/unified_eval.csv",
        "--out-report", "error_analysis/unified_eval_report.txt",
    ],
    check=True,
)
