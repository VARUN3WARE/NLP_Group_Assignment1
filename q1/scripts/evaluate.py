#!/usr/bin/env python3
"""Run the full Question 1 evaluation suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/evaluate.py` from q1/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segpos.eval.experiments import run_full_evaluation


def main() -> None:
    run_full_evaluation()


if __name__ == "__main__":
    main()
