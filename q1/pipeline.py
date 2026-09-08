"""Shim: old `from pipeline import ...` → `segpos.pipeline`."""

from segpos.pipeline import (
    DEFAULT_ALPHA,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_BETA,
    DEFAULT_BUNDLE_PATH,
    DEFAULT_MAX_WORD_LENGTH,
    EnglishPipeline,
    load_english_pipeline,
    train_english_pipeline,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BEAM_WIDTH",
    "DEFAULT_BETA",
    "DEFAULT_BUNDLE_PATH",
    "DEFAULT_MAX_WORD_LENGTH",
    "EnglishPipeline",
    "load_english_pipeline",
    "train_english_pipeline",
]
