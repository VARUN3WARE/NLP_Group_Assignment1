"""
Make Q1 and Q3 importable from anywhere in the monorepo.

Usage from q4 (or any sibling folder):

    from q1_paths import ensure_q1_on_path, ensure_q3_on_path
    ensure_q1_on_path()
    from segpos import load_english_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
Q1_ROOT = REPO_ROOT / "q1"
Q3_ROOT = REPO_ROOT / "q3"


def ensure_q1_on_path() -> Path:
    path = str(Q1_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return Q1_ROOT


def ensure_q3_on_path() -> Path:
    path = str(Q3_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return Q3_ROOT


def ensure_repo_on_path() -> Path:
    path = str(REPO_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return REPO_ROOT
