"""
English segmentation + POS pipeline with train / save / load for Q4 reuse.

Do not retrain inside Question 4 — call `load_english_pipeline()` instead.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from corpus import load_brown, split_brown
from joint_decoder import JointBeamDecoder
from language_model import TrigramLanguageModel
from pos_tagger import TrigramPOSTagger
from segmentation import ViterbiSegmenter

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_BUNDLE_PATH = DEFAULT_MODEL_DIR / "english_pipeline.pkl"

# Defaults selected for Q4 live checks (documented in the Q4 report).
DEFAULT_MAX_WORD_LENGTH = 20
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_BEAM_WIDTH = 8


@dataclass
class EnglishPipeline:
    lm: TrigramLanguageModel
    tagger: TrigramPOSTagger
    max_word_length: int = DEFAULT_MAX_WORD_LENGTH
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    beam_width: int = DEFAULT_BEAM_WIDTH

    def __post_init__(self):
        self.segmenter = ViterbiSegmenter(self.lm, max_word_length=self.max_word_length)
        self.decoder = JointBeamDecoder(
            self.lm,
            self.tagger,
            max_word_length=self.max_word_length,
            alpha=self.alpha,
            beta=self.beta,
            beam_width=self.beam_width,
        )

    @property
    def vocabulary(self):
        return self.lm.vocabulary

    def segment(self, text: str):
        return self.segmenter.segment(text)

    def tag(self, words):
        return self.tagger.tag(words)

    def decode(self, text: str):
        """Joint beam segmentation + POS (preferred for Q4)."""
        return self.decoder.decode(text)

    def should_split(self, token: str):
        return self.decoder.should_split(token)

    def segment_then_tag(self, text: str):
        """Fallback pipeline: Viterbi segment, then Viterbi tag."""
        words = self.segment(text)
        return self.tag(words)

    def save(self, path=DEFAULT_BUNDLE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lm": self.lm,
            "tagger": self.tagger,
            "max_word_length": self.max_word_length,
            "alpha": self.alpha,
            "beta": self.beta,
            "beam_width": self.beam_width,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path


def train_english_pipeline(
    max_word_length: int = DEFAULT_MAX_WORD_LENGTH,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> EnglishPipeline:
    """Train English LM + POS tagger on Brown 80% split."""
    sentences = load_brown()
    train_sentences, _ = split_brown(sentences)
    train_words = [[word for word, _tag in sent] for sent in train_sentences]

    lm = TrigramLanguageModel()
    lm.train(train_words)

    tagger = TrigramPOSTagger()
    tagger.train(train_sentences)

    return EnglishPipeline(
        lm=lm,
        tagger=tagger,
        max_word_length=max_word_length,
        alpha=alpha,
        beta=beta,
        beam_width=beam_width,
    )


def load_english_pipeline(path=DEFAULT_BUNDLE_PATH) -> EnglishPipeline:
    """Load a previously trained English pipeline (no retraining)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No English pipeline at {path}. Run from q1/:  python main.py train-english"
        )
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    return EnglishPipeline(
        lm=payload["lm"],
        tagger=payload["tagger"],
        max_word_length=payload.get("max_word_length", DEFAULT_MAX_WORD_LENGTH),
        alpha=payload.get("alpha", DEFAULT_ALPHA),
        beta=payload.get("beta", DEFAULT_BETA),
        beam_width=payload.get("beam_width", DEFAULT_BEAM_WIDTH),
    )
