"""Shared filesystem paths for Question 4."""

from pathlib import Path

Q4_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Q4_ROOT / "artifacts"
RESULTS_DIR = Q4_ROOT / "results"

BIGRAM_PATH = ARTIFACTS_DIR / "q4_bigram.pkl"
TRIGRAM_PATH = ARTIFACTS_DIR / "q4_trigram.pkl"
PCFG_PATH = ARTIFACTS_DIR / "q4_pcfg.pkl"
