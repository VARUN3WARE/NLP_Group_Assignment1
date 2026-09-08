"""Shared filesystem paths for Question 1."""

from pathlib import Path

# q1/  (package parent)
Q1_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = Q1_ROOT / "artifacts"
DATA_DIR = Q1_ROOT / "data"
DEFAULT_SPANISH_DIR = DATA_DIR / "spanish" / "UD_Spanish-GSD"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "english_pipeline.pkl"
