"""Question 1 package: word segmentation + POS tagging (English / Spanish)."""

from segpos.pipeline import (
    DEFAULT_BUNDLE_PATH,
    EnglishPipeline,
    load_english_pipeline,
    train_english_pipeline,
)
from segpos.lm.trigram import TrigramLanguageModel
from segpos.segmentation.viterbi import ViterbiSegmenter
from segpos.segmentation.joint_beam import JointBeamDecoder
from segpos.tagging.pos_tagger import MorphologyAwarePOSTagger, TrigramPOSTagger
from segpos.baselines.simple import GreedyLongestMatchSegmenter, MostFrequentTagger

__all__ = [
    "DEFAULT_BUNDLE_PATH",
    "EnglishPipeline",
    "load_english_pipeline",
    "train_english_pipeline",
    "TrigramLanguageModel",
    "ViterbiSegmenter",
    "JointBeamDecoder",
    "TrigramPOSTagger",
    "MorphologyAwarePOSTagger",
    "GreedyLongestMatchSegmenter",
    "MostFrequentTagger",
]
