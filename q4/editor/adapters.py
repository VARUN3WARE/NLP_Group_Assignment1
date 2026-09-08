"""
Load trained Q1 / Q3 components once. Never retrain inside Q4.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Repo root (parent of q4/) on path for q1_paths
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from q1_paths import ensure_q1_on_path, ensure_q3_on_path


@lru_cache(maxsize=1)
def get_q1_pipeline():
    """English joint beam decoder + LM + POS from Question 1."""
    ensure_q1_on_path()
    from segpos import load_english_pipeline

    return load_english_pipeline()


@lru_cache(maxsize=1)
def get_q3_models(method: str = "B", real_word_threshold: float = 5.0):
    """Vocabulary / corrector from Question 3 (Method B by default)."""
    ensure_q3_on_path()
    from spelling import load_models

    return load_models(method=method, real_word_threshold=real_word_threshold)


def q3_vocabulary():
    model, _ = get_q3_models()
    return model.vocabulary


def q3_corrector():
    _, corrector = get_q3_models()
    return corrector
