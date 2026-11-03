"""
Training entry point (supervisor requirement: one training script).

Runs every stage that fits something, in order, for all three targets:
labels (causal HMM fit on 2000-2015) -> features -> supervised models (XGBoost,
logistic regression) -> Isolation Forest benchmark. Stops at the first error.
Data are read from the committed cache only; nothing is downloaded.

Usage (from the repo root):
    python code/train.py                 # all targets
    python code/train.py --index GSPC    # one target
"""

import argparse
import subprocess
import sys

STAGES = ["labelling.py", "features.py", "baseline_models.py", "isolation_forest.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=["GSPC", "N225", "FTSE"], help="default: all three")
    args = parser.parse_args()
    indices = [args.index] if args.index else ["GSPC", "N225", "FTSE"]
    for stage in STAGES:
        for idx in indices:
            print(f"=== {stage} --index {idx} ===", flush=True)
            subprocess.run([sys.executable, f"code/{stage}", "--index", idx], check=True)
    print("=== train.py finished ===")


if __name__ == "__main__":
    main()
