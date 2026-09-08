"""Shared filesystem paths for Question 3."""

from pathlib import Path

Q3_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Q3_ROOT / "artifacts"
RESULTS_DIR = Q3_ROOT / "results"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "brown_lm.pkl"
DEFAULT_SPLIT_PATH = ARTIFACTS_DIR / "test_split.pkl"
