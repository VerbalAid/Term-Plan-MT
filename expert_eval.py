"""
TermPlanMT Expert Medical Translation Evaluation — thin wrapper around eval_pipeline.py

Original pilot: 25 segments, seed=99, all 7 system variants (both NER conditions).
Superseded by unified_eval.py (30 segments, seed=42, same IDs as audit).
Kept for reproducibility of the pilot results in error_analysis/expert_eval.csv.
"""

import subprocess, sys

subprocess.run(
    [
        sys.executable, "eval_pipeline.py",
        "--segments",   "25",
        "--seed",       "99",
        "--systems",    "s1,s2,s3,s3ft,s5b,s5ft,s6",
        "--out-csv",    "error_analysis/expert_eval.csv",
        "--out-report", "error_analysis/expert_report.txt",
    ],
    check=True,
)
