"""Q4-owned add-k smoothed bigram and trigram LMs (Brown). Separate from Q1/Q3."""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from editor.paths import ARTIFACTS_DIR, BIGRAM_PATH, TRIGRAM_PATH

BOS = "<s>"
EOS = "</s>"
_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")


def _norm_tokens(tokens: Iterable[str]) -> list[str]:
    out = []
    for t in tokens:
        t = t.lower()
        if _WORD_RE.match(t):
            out.append(t)
    return out


@dataclass
class AddKNgramLM:
    """Add-k smoothed n-gram language model (n=2 or n=3)."""

    order: int
    k: float = 0.01
    ngram_counts: Counter = field(default_factory=Counter)
    context_counts: Counter = field(default_factory=Counter)
    vocabulary: frozenset = frozenset()
    total_tokens: int = 0

    @classmethod
    def train(
        cls,
        sentences: Sequence[Sequence[str]],
        order: int,
        k: float = 0.01,
    ) -> "AddKNgramLM":
        assert order in (2, 3)
        model = cls(order=order, k=k)
        vocab: set[str] = set()
        total = 0

        for raw in sentences:
            sent = _norm_tokens(raw)
            if not sent:
                continue
            total += len(sent)
            vocab.update(sent)
            padded = [BOS] * (order - 1) + sent + [EOS]
            for i in range(order - 1, len(padded)):
                ngram = tuple(padded[i - order + 1 : i + 1])
                context = ngram[:-1]
                model.ngram_counts[ngram] += 1
                model.context_counts[context] += 1

        model.vocabulary = frozenset(vocab)
        model.total_tokens = total
        return model

    @property
    def vocab_size(self) -> int:
        # + boundary symbols mass
        return len(self.vocabulary) + 2

    def log_prob(self, context: tuple[str, ...], word: str) -> float:
        numer = self.ngram_counts.get(context + (word,), 0) + self.k
        denom = self.context_counts.get(context, 0) + self.k * self.vocab_size
        return math.log(numer / denom)

    def sentence_log_prob(self, tokens: Sequence[str]) -> float:
        toks = _norm_tokens(tokens)
        padded = [BOS] * (self.order - 1) + toks + [EOS]
        score = 0.0
        for i in range(self.order - 1, len(padded)):
            context = tuple(padded[i - self.order + 1 : i])
            score += self.log_prob(context, padded[i])
        return score

    def perplexity(self, tokens: Sequence[str]) -> float:
        toks = _norm_tokens(tokens)
        n = len(toks) + 1  # count </s>
        if n <= 0:
            return float("inf")
        return math.exp(-self.sentence_log_prob(toks) / n)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: Path) -> "AddKNgramLM":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def __repr__(self) -> str:
        return (
            f"AddKNgramLM(order={self.order}, types={len(self.vocabulary):,}, "
            f"tokens={self.total_tokens:,}, k={self.k})"
        )


def train_q4_language_models(k: float = 0.01) -> tuple[AddKNgramLM, AddKNgramLM]:
    """Train bigram + trigram on Brown; save under q4/artifacts/."""
    import nltk

    nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    sentences = list(brown.sents())
    bigram = AddKNgramLM.train(sentences, order=2, k=k)
    trigram = AddKNgramLM.train(sentences, order=3, k=k)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bigram.save(BIGRAM_PATH)
    trigram.save(TRIGRAM_PATH)
    return bigram, trigram


def load_q4_language_models(k: float = 0.01) -> tuple[AddKNgramLM, AddKNgramLM]:
    if BIGRAM_PATH.exists() and TRIGRAM_PATH.exists():
        return AddKNgramLM.load(BIGRAM_PATH), AddKNgramLM.load(TRIGRAM_PATH)
    return train_q4_language_models(k=k)
