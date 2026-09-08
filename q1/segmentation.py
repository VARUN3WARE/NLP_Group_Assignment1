"""Shim: old `from segmentation import ...` → `segpos.segmentation.viterbi`."""

from segpos.segmentation.viterbi import ViterbiSegmenter

__all__ = ["ViterbiSegmenter"]
