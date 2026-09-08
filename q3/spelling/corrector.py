"""Spelling correction logic (non-word + real-word)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

from spelling.candidates import EditDistance1Generator, SymmetricDeleteIndex
from spelling.language_model import LanguageModel
from spelling.paths import DEFAULT_MODEL_PATH
from spelling.tokenize import BOS, EOS, NUM


Method = Literal["A", "B", "both"]
Scoring = Literal["add-k", "interpolated"]

_TEXT_SPLIT_RE = re.compile(r"([A-Za-z]+(?:'[A-Za-z]+)?)")


@dataclass
class Correction:
    """One change the corrector decided to make."""

    index: int
    original: str
    corrected: str
    kind: Literal["non-word", "real-word"]
    score_gain: float


class SpellingCorrector:
    """Edit-distance-1 spelling corrector over a `LanguageModel`."""

    def __init__(
        self,
        model: LanguageModel,
        method: Method = "both",
        real_word_threshold: float = 5.0,
        min_real_word_length: int = 3,
        real_word_max_freq_rank: int | None = None,
        scoring: Scoring = "interpolated",
        unigram_discount: float = 0.0,
    ):
        """
        Args:
            model: trained vocabulary / unigram / bigram model.
            method: which candidate generator(s) to use -- "A", "B", or "both"
                (the union, as the specification asks for).  A and B return
                identical sets, so this only changes runtime, never output.
            real_word_threshold: how many nats of bigram log-probability a
                candidate must beat the typed word by before it is proposed.
                This is the noisy-channel prior on "the user did not make a
                mistake": threshold = log((1 - eps) * N / eps) for an error
                rate eps over roughly N edit-1 neighbours.
            min_real_word_length: never second-guess very short words; "a",
                "I", "of" have dozens of neighbours and almost no signal.
            real_word_max_freq_rank: if set, only words *outside* the N most
                frequent types are eligible for real-word correction.  Off by
                default: measured against the threshold at a matched
                false-alarm rate it is a strictly worse trade (see REPORT.md).
            scoring: which bigram estimate scores the local phrase --
                "add-k" (exactly the Part 1 model) or "interpolated"
                (Jelinek-Mercer back-off to the unigram, the better decision
                rule and the default).
            unigram_discount: beta in
                score = logP(x|prev) + logP(next|x) - beta * logP(x).
                beta = 0 (default) is the phrase probability the brief asks
                for.  beta = 1 turns the score into a pure PMI "contextual
                fit" measure that ignores how common the candidate is; that
                removes the "brown fox -> brown for" failure mode but scores
                worse on the synthetic test set, because randomly corrupting a
                word almost always makes it rarer, so a frequency prior gets
                rewarded there.  See REPORT.md.
        """
        self.model = model
        self.method = method
        self.scoring = scoring
        self.real_word_threshold = real_word_threshold
        self.min_real_word_length = min_real_word_length
        self.unigram_discount = unigram_discount

        self.method_a = EditDistance1Generator(model.vocabulary)
        self.method_b = SymmetricDeleteIndex(model.vocabulary, verify=True)

        self._protected: frozenset[str] = frozenset()
        if real_word_max_freq_rank:
            self._protected = frozenset(
                w for w, _ in model.unigram_counts.most_common(real_word_max_freq_rank)
            )

    def candidates(self, word: str, method: Method | None = None) -> set[str]:
        """Vocabulary words within one edit of `word`, via the chosen method."""
        method = method or self.method
        if method == "A":
            return self.method_a.candidates(word)
        if method == "B":
            return self.method_b.candidates(word)
        return self.method_a.candidates(word) | self.method_b.candidates(word)

    def correct_non_word(self, word: str, method: Method | None = None) -> str:
        """Best edit-distance-1 replacement for an out-of-vocabulary token.

        Returns `word` unchanged when nothing sits within one edit -- the
        specification scopes this corrector to edit distance 1, so an
        unreachable word is reported rather than guessed at.
        """
        cands = self.candidates(word, method)
        if not cands:
            return word
        counts = self.model.unigram_counts
        return max(cands, key=lambda c: (counts[c], -len(c), c))

    def _log_bigram(self, prev: str, word: str) -> float:
        if self.scoring == "add-k":
            return self.model.log_bigram_prob(prev, word)
        return self.model.log_bigram_interp(prev, word)

    def _phrase_log_prob(self, prev: str, word: str, nxt: str) -> float:
        """log P(word | prev) + log P(next | word) -- the local phrase score.

        This is exactly "the probability of the phrase" the brief asks for:
        the candidate is scored against both of its neighbours, so a word can
        only win by fitting the context on both sides.
        """
        score = self._log_bigram(prev, word) + self._log_bigram(word, nxt)
        if self.unigram_discount:
            score -= self.unigram_discount * self.model.log_unigram_prob(word)
        return score

    def correct_real_word(
        self,
        word: str,
        prev: str = BOS,
        nxt: str = EOS,
        method: Method | None = None,
    ) -> tuple[str, float]:
        """Context-check an in-vocabulary word.

        Returns (word_or_correction, log-probability margin).  The margin is
        0.0 when the word is left alone.
        """
        if len(word) < self.min_real_word_length or word in self._protected:
            return word, 0.0

        cands = self.candidates(word, method)
        if not cands:
            return word, 0.0

        base = self._phrase_log_prob(prev, word, nxt)
        best, best_score = word, base
        for cand in cands:
            score = self._phrase_log_prob(prev, cand, nxt)
            if score > best_score:
                best, best_score = cand, score

        gain = best_score - base
        if best != word and gain >= self.real_word_threshold:
            return best, gain
        return word, 0.0

    def correct_tokens(
        self,
        tokens: Sequence[str],
        method: Method | None = None,
        check_real_words: bool = True,
    ) -> tuple[list[str], list[Correction]]:
        """Correct a normalised word sequence, left to right.

        Non-word errors are always fixed.  Real-word checking uses the
        *already corrected* left context and the raw right context (the right
        neighbour has not been decided yet).
        """
        out = list(tokens)
        changes: list[Correction] = []
        vocab = self.model.vocabulary

        for i, word in enumerate(tokens):
            if word in (NUM, BOS, EOS):
                continue
            prev = out[i - 1] if i > 0 else BOS
            nxt = tokens[i + 1] if i + 1 < len(tokens) else EOS

            if word not in vocab:
                fixed = self.correct_non_word(word, method)
                if fixed != word:
                    out[i] = fixed
                    changes.append(Correction(i, word, fixed, "non-word", 0.0))
            elif check_real_words:
                fixed, gain = self.correct_real_word(word, prev, nxt, method)
                if fixed != word:
                    out[i] = fixed
                    changes.append(Correction(i, word, fixed, "real-word", gain))

        return out, changes

    def correct_text(
        self,
        text: str,
        method: Method | None = None,
        check_real_words: bool = True,
    ) -> tuple[str, list[Correction], list[tuple[str, str]]]:
        """Correct raw text, preserving punctuation, spacing and casing.

        Returns (corrected_text, corrections, pieces) where `pieces` is the
        rebuilt (original_chunk, corrected_chunk) stream that the CLI uses to
        highlight exactly what changed.
        """
        chunks = _TEXT_SPLIT_RE.split(text)
        word_positions = [i for i in range(1, len(chunks), 2)]
        words = [chunks[i].lower() for i in word_positions]

        corrected, changes = self.correct_tokens(
            words, method=method, check_real_words=check_real_words
        )

        out_chunks = list(chunks)
        for pos, new_word in zip(word_positions, corrected):
            out_chunks[pos] = _match_case(chunks[pos], new_word)
        pieces = list(zip(chunks, out_chunks))

        return "".join(out_chunks), changes, pieces


def _match_case(original: str, replacement: str) -> str:
    """Re-apply the original token's capitalisation to the correction."""
    if original == replacement:
        return original
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement

@lru_cache(maxsize=4)
def load_models(
    path: str | Path = DEFAULT_MODEL_PATH,
    method: str = "both",
    real_word_threshold: float = 5.0,
) -> tuple[LanguageModel, SpellingCorrector]:
    """Return (language model, ready-to-use corrector).

    Raises a helpful error if `build_models.py` has not been run yet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run from q3/:  python main.py train"
        )
    model = LanguageModel.load(path)
    corrector = SpellingCorrector(
        model, method=method, real_word_threshold=real_word_threshold
    )
    return model, corrector

