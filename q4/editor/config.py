"""Q4 configuration knobs (justified in REPORT.md)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditorConfig:
    # Part 1 — typing simulation
    merge_probability: float = 0.08
    grammar_trigger_n: int = 5
    typing_delay_s: float = 0.05
    long_token_chars: int = 12

    # Q4-owned grammar LMs (separate from Q1/Q3 artifacts)
    add_k: float = 0.01
    grammar_perplexity_threshold: float = 5000.0
    real_word_threshold: float = 5.0

    # Decision rule (Part 4)
    pcfg_outlier_z: float = 2.5

    # Passage sampling
    min_sentences: int = 5
    max_sentences: int = 8


DEFAULT_CONFIG = EditorConfig()
