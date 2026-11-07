"""
Evaluation entry point (supervisor requirement: one evaluation script).

Run after code/train.py. Uses the saved test predictions and result files to run the
statistical tests and explanations and to build every table and figure:
Diebold-Mariano tests -> Wilcoxon / Friedman tests -> global SHAP -> figures ->
report tables. Stops at the first error.

Usage (from the repo root):
    python code/test.py
"""

import subprocess
import sys

STAGES = ["dm_tests.py", "significance.py", "shap_analysis.py", "figures.py", "macro_range.py", "event_tables.py", "report_tables.py"]


def main():
    for stage in STAGES:
        print(f"=== {stage} ===", flush=True)
        subprocess.run([sys.executable, f"code/{stage}"], check=True)
    print("=== test.py finished ===")


if __name__ == "__main__":
    main()
