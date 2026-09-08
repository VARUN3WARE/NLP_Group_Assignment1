"""
Question 3 - Building, Benchmarking and Deploying an Efficient Spelling Corrector.

Everything the corrector needs, in one module:

    Part 1  corpus preparation   `LanguageModel`      vocabulary, unigram, bigram
    Part 2  candidate generation `EditDistance1Generator` (Method A)
                                 `SymmetricDeleteIndex`   (Method B, SymSpell)
    Part 3  correction logic     `SpellingCorrector`  non-word + real-word
    Part 4  evaluation           `build_test_sets`, `evaluate`, `speed_demon`

Question 4 reuses the trained artefacts without retraining them:

    from spelling import load_models
    model, corrector = load_models()
    corrector.correct_text("This is a test sentnce.")

Run `python main.py --help` for the drivers.
"""

from __future__ import annotations

import math
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "brown_lm.pkl"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "models" / "test_split.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"


BOS = "<s>"
EOS = "</s>"
NUM = "<num>"

_WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
_HAS_DIGIT_RE = re.compile(r"\d")

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


def normalise(token: str) -> str | None:
    """Map a raw corpus token to its model form.

    Returns the lower-cased word, the ``<num>`` placeholder for numeric
    tokens, or ``None`` for pure punctuation (which is dropped).
    """
    token = token.lower()
    if _WORD_RE.match(token):
        return token
    if _HAS_DIGIT_RE.search(token):
        return NUM
    return None


def normalise_sentence(tokens: Iterable[str]) -> list[str]:
    """Normalise a sentence, dropping tokens that carry no lexical content."""
    out = []
    for tok in tokens:
        norm = normalise(tok)
        if norm is not None:
            out.append(norm)
    return out


@dataclass
class LanguageModel:
    """Vocabulary + unigram frequencies + add-k smoothed bigram model."""

    unigram_counts: Counter = field(default_factory=Counter)
    bigram_counts: Counter = field(default_factory=Counter)
    context_counts: Counter = field(default_factory=Counter)
    k: float = 0.01
    lam: float = 0.85
    unigram_k: float = 0.5

    total_tokens: int = 0
    vocabulary: frozenset = frozenset()

    @classmethod
    def train(cls, sentences: Iterable[Sequence[str]], k: float = 0.01) -> "LanguageModel":
        """Train the unigram + bigram models from an iterable of sentences.

        Each sentence is a sequence of raw (untokenised-case) word strings.
        """
        model = cls(k=k)
        unigrams = model.unigram_counts
        bigrams = model.bigram_counts
        contexts = model.context_counts

        for raw_sent in sentences:
            sent = normalise_sentence(raw_sent)
            if not sent:
                continue
            unigrams.update(sent)
            padded = [BOS] + sent + [EOS]
            for left, right in zip(padded, padded[1:]):
                bigrams[(left, right)] += 1
                contexts[left] += 1

        model.total_tokens = sum(unigrams.values())
        model.vocabulary = frozenset(w for w in unigrams if w != NUM)
        return model

    def unigram_prob(self, word: str) -> float:
        """P(word) = C(word) / N.  Unknown words get 0."""
        return self.unigram_counts.get(word, 0) / self.total_tokens

    def log_unigram_prob(self, word: str, k: float | None = None) -> float:
        """Add-k smoothed log P(word); never -inf so it is safe to sum."""
        k = self.k if k is None else k
        v = len(self.vocabulary) + 1
        return math.log(
            (self.unigram_counts.get(word, 0) + k) / (self.total_tokens + k * v)
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary) + 2

    def bigram_prob(self, prev: str, word: str) -> float:
        """Add-k smoothed P(word | prev)."""
        numer = self.bigram_counts.get((prev, word), 0) + self.k
        denom = self.context_counts.get(prev, 0) + self.k * self.vocab_size
        return numer / denom

    def log_bigram_prob(self, prev: str, word: str) -> float:
        return math.log(self.bigram_prob(prev, word))

    def bigram_prob_interp(self, prev: str, word: str) -> float:
        """P(word | prev) interpolated with the unigram distribution."""
        p_uni = ((self.unigram_counts.get(word, 0) + self.unigram_k)
                 / (self.total_tokens + self.unigram_k * self.vocab_size))
        context = self.context_counts.get(prev, 0)
        p_ml = self.bigram_counts.get((prev, word), 0) / context if context else 0.0
        return self.lam * p_ml + (1.0 - self.lam) * p_uni

    def log_bigram_interp(self, prev: str, word: str) -> float:
        return math.log(self.bigram_prob_interp(prev, word))

    def sentence_log_prob(self, tokens: Sequence[str]) -> float:
        """Log P(sentence) under the bigram model, with <s>/</s> padding."""
        padded = [BOS] + list(tokens) + [EOS]
        return sum(
            self.log_bigram_prob(left, right)
            for left, right in zip(padded, padded[1:])
        )

    def perplexity(self, tokens: Sequence[str]) -> float:
        n = len(tokens) + 1
        if n == 0:
            return float("inf")
        return math.exp(-self.sentence_log_prob(tokens) / n)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "LanguageModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def __repr__(self) -> str:
        return (
            f"LanguageModel(types={len(self.vocabulary):,}, "
            f"tokens={self.total_tokens:,}, bigrams={len(self.bigram_counts):,}, "
            f"k={self.k})"
        )


def brown_sentences() -> list[list[str]]:
    """All Brown corpus sentences as lists of raw tokens."""
    from nltk.corpus import brown

    return [list(sent) for sent in brown.sents()]


def train_test_split(sentences: Sequence[Sequence[str]], test_ratio: float = 0.10,
                     seed: int = 42):
    """Hold out `test_ratio` of the sentences for the Part 4 evaluation.

    The corrector is built from the *training* portion only, so the accuracy
    numbers in Part 4 are not inflated by having memorised the test sentences.
    """
    import random

    idx = list(range(len(sentences)))
    random.Random(seed).shuffle(idx)
    n_test = int(round(len(idx) * test_ratio))
    test_idx = set(idx[:n_test])
    train = [s for i, s in enumerate(sentences) if i not in test_idx]
    test = [sentences[i] for i in idx[:n_test]]
    return train, test


def deletes1(word: str) -> Set[str]:
    """Every string obtained by deleting exactly one character of `word`."""
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def within_edit_distance_1(a: str, b: str) -> bool:
    """True iff `a` and `b` are within Damerau-Levenshtein distance 1.

    Written as an explicit length-case analysis rather than a full DP table:
    at distance <= 1 only four shapes are possible, and each can be checked in
    a single linear scan.  This is the verification step that turns SymSpell's
    *approximate* bucket lookup into an exact edit-distance-1 result set.
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return j == i + 1 and a[i] == b[j] and a[j] == b[i]
        return False

    short, long = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


class EditDistance1Generator:
    """Method A: enumerate all edit-distance-1 strings, keep the real words."""

    def __init__(self, vocabulary: Iterable[str], alphabet: str = ALPHABET):
        self.vocabulary = frozenset(vocabulary)
        self.alphabet = alphabet

    def edits1(self, word: str) -> Set[str]:
        """All strings one edit away from `word` (not filtered by vocabulary).

        For a word of length n this produces
            n deletions + (n-1) transpositions + n*|A| replacements
            + (n+1)*|A| insertions
        candidate strings, i.e. O(n * |alphabet|).
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = {L + R[1:] for L, R in splits if R}
        transposes = {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1}
        replaces = {L + c + R[1:] for L, R in splits if R for c in self.alphabet}
        inserts = {L + c + R for L, R in splits for c in self.alphabet}
        return deletes | transposes | replaces | inserts

    def candidates(self, word: str) -> Set[str]:
        """Vocabulary words within one edit of `word` (excluding `word`)."""
        return {w for w in self.edits1(word) if w in self.vocabulary} - {word}

    def __repr__(self) -> str:
        return f"EditDistance1Generator(|V|={len(self.vocabulary):,})"


class SymmetricDeleteIndex:
    """Method B: SymSpell.  All the work happens once, at build time.

    The index maps a *deletion variant* to the vocabulary words that produce
    it.  Every word is also indexed under itself, which is what lets the
    lookup catch the "the query has one extra character" case.

    Why one deletion on each side is enough for edit distance 1:
      * substitution  - delete the changed position on both sides -> equal
      * transposition - delete one of the two swapped chars on each side
      * insertion in the query  - the query's deletion equals the word
      * deletion in the query   - the query equals one of the word's deletions
    """

    def __init__(self, vocabulary: Iterable[str], verify: bool = True):
        self.vocabulary = frozenset(vocabulary)
        self.verify = verify
        self.index: dict[str, list[str]] = self._build_index(self.vocabulary)

    @staticmethod
    def _build_index(vocabulary: Iterable[str]) -> dict[str, list[str]]:
        index: dict[str, list[str]] = defaultdict(list)
        for word in vocabulary:
            index[word].append(word)
            for variant in deletes1(word):
                index[variant].append(word)
        return dict(index)

    def candidates(self, word: str) -> Set[str]:
        """Vocabulary words within one edit of `word` (excluding `word`)."""
        index = self.index
        found: Set[str] = set()

        for probe in (word, *deletes1(word)):
            hits = index.get(probe)
            if hits:
                found.update(hits)

        found.discard(word)
        if self.verify:
            found = {w for w in found if within_edit_distance_1(word, w)}
        return found

    @property
    def n_entries(self) -> int:
        """Number of distinct keys stored in the pre-computed dictionary."""
        return len(self.index)

    @property
    def n_postings(self) -> int:
        """Total (key -> word) pairs, i.e. the real memory cost of Method B."""
        return sum(len(v) for v in self.index.values())

    def __repr__(self) -> str:
        return (
            f"SymmetricDeleteIndex(|V|={len(self.vocabulary):,}, "
            f"keys={self.n_entries:,}, postings={self.n_postings:,}, "
            f"verify={self.verify})"
        )


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


_CORRUPTION_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def single_edits(word: str, alphabet: str = _CORRUPTION_ALPHABET) -> list[str]:
    """Every distinct string one edit away from `word`, as a list."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    out = set()
    out.update(L + R[1:] for L, R in splits if R)
    out.update(L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1)
    out.update(L + c + R[1:] for L, R in splits if R for c in alphabet)
    out.update(L + c + R for L, R in splits for c in alphabet)
    out.discard(word)
    return sorted(out)


def corrupt(word: str, vocabulary: frozenset, rng: random.Random,
            want_real_word: bool) -> str | None:
    """Apply one random edit to `word`.

    `want_real_word` selects between an edit that lands on another vocabulary
    word (a real-word error) and one that lands outside the vocabulary (a
    non-word error).  Returns None when no such edit exists.
    """
    pool = [e for e in single_edits(word)
            if (e in vocabulary) == want_real_word and e]
    if not pool:
        return None
    return rng.choice(pool)


@dataclass
class TestCase:
    """One corrupted sentence."""

    tokens: Sequence[str]
    index: int
    original: str
    corrupted: str
    noisy: list[str]


def build_test_sets(
    test_sentences: Iterable[Sequence[str]],
    vocabulary: frozenset,
    seed: int = 7,
    min_word_length: int = 3,
    max_cases: int | None = None,
) -> tuple[list[TestCase], list[TestCase]]:
    """Build the paired non-word and real-word test sets.

    Only sentences where BOTH a non-word and a real-word corruption of the
    same target word exist are kept, so the two sets are aligned.
    """
    rng = random.Random(seed)
    non_word: list[TestCase] = []
    real_word: list[TestCase] = []

    for raw in test_sentences:
        tokens = normalise_sentence(raw)
        targets = [i for i, w in enumerate(tokens)
                   if w != NUM and len(w) >= min_word_length and w in vocabulary]
        if not targets:
            continue
        rng.shuffle(targets)

        for idx in targets:
            word = tokens[idx]
            nw = corrupt(word, vocabulary, rng, want_real_word=False)
            rw = corrupt(word, vocabulary, rng, want_real_word=True)
            if nw is None or rw is None:
                continue

            for corrupted, bucket in ((nw, non_word), (rw, real_word)):
                noisy = list(tokens)
                noisy[idx] = corrupted
                bucket.append(TestCase(tokens, idx, word, corrupted, noisy))
            break

        if max_cases and len(non_word) >= max_cases:
            break

    return non_word, real_word


@dataclass
class AccuracyResult:
    n: int
    target_correct: int
    sentence_exact: int
    untouched: int
    wrong_correction: int
    collateral: int
    elapsed: float

    @property
    def accuracy(self) -> float:
        return self.target_correct / self.n if self.n else 0.0

    @property
    def sentence_accuracy(self) -> float:
        return self.sentence_exact / self.n if self.n else 0.0

    def as_row(self, label: str) -> str:
        return (
            f"{label:<22}{self.n:>7}{self.accuracy:>11.2%}"
            f"{self.sentence_accuracy:>12.2%}{self.untouched:>11}"
            f"{self.wrong_correction:>11}{self.collateral:>12}"
            f"{self.elapsed:>10.1f}s"
        )


def evaluate(
    corrector: SpellingCorrector,
    cases: Sequence[TestCase],
    check_real_words: bool,
    method: str | None = None,
) -> AccuracyResult:
    """Run the corrector over a test set and score it."""
    target_correct = sentence_exact = untouched = wrong = collateral = 0
    t0 = time.perf_counter()

    for case in cases:
        out, _ = corrector.correct_tokens(
            case.noisy, method=method, check_real_words=check_real_words
        )
        got = out[case.index]
        if got == case.original:
            target_correct += 1
        elif got == case.corrupted:
            untouched += 1
        else:
            wrong += 1
        if out == list(case.tokens):
            sentence_exact += 1
        if any(o != g for i, (o, g) in enumerate(zip(out, case.tokens))
               if i != case.index):
            collateral += 1

    return AccuracyResult(
        n=len(cases),
        target_correct=target_correct,
        sentence_exact=sentence_exact,
        untouched=untouched,
        wrong_correction=wrong,
        collateral=collateral,
        elapsed=time.perf_counter() - t0,
    )


@dataclass
class FalseAlarmResult:
    """What the corrector does to text that was already correct."""

    n_sentences: int
    n_words: int
    real_word_changes: int
    non_word_changes: int
    damaged_sentences: int

    @property
    def rate_per_word(self) -> float:
        return self.real_word_changes / self.n_words if self.n_words else 0.0

    @property
    def rate_per_sentence(self) -> float:
        return self.damaged_sentences / self.n_sentences if self.n_sentences else 0.0


def false_alarm_rate(
    corrector: SpellingCorrector,
    cases: Sequence[TestCase],
    method: str | None = None,
) -> FalseAlarmResult:
    """How often does real-word checking damage an already-correct sentence?

    Runs the corrector over the *uncorrupted* gold sentences.  Rewrites are
    split by kind: only real-word rewrites are false alarms.  A rewrite of an
    out-of-vocabulary word is the corrector doing its job on a word the
    training split genuinely never saw, which is a different phenomenon.
    """
    rw = nw = damaged = words = 0
    for case in cases:
        _, changes = corrector.correct_tokens(
            case.tokens, method=method, check_real_words=True
        )
        rw_here = sum(1 for c in changes if c.kind == "real-word")
        rw += rw_here
        nw += len(changes) - rw_here
        damaged += 1 if rw_here else 0
        words += len(case.tokens)
    return FalseAlarmResult(len(cases), words, rw, nw, damaged)


def threshold_sweep(
    model: LanguageModel,
    real_word_cases: Sequence[TestCase],
    clean_cases: Sequence[TestCase],
    thresholds: Sequence[float] = (1, 2, 3, 4, 5, 6, 8, 10),
    scoring: str = "interpolated",
    unigram_discount: float = 0.0,
    method: str = "B",
) -> list[dict]:
    """Accuracy / false-alarm trade-off as the "significantly higher" bar moves.

    This is the experiment behind the chosen default threshold: there is no
    free lunch, only an operating point.
    """
    rows = []
    for thr in thresholds:
        corrector = SpellingCorrector(
            model, method=method, real_word_threshold=thr,
            scoring=scoring, unigram_discount=unigram_discount,
        )
        acc = evaluate(corrector, real_word_cases, check_real_words=True)
        fa = false_alarm_rate(corrector, clean_cases)
        rows.append({
            "threshold": thr,
            "real_word_accuracy": acc.accuracy,
            "false_alarm_per_word": fa.rate_per_word,
            "false_alarm_per_sentence": fa.rate_per_sentence,
        })
    return rows


@dataclass
class BenchmarkResult:
    label: str
    n_words: int
    elapsed: float
    n_candidates: int

    @property
    def per_word_ms(self) -> float:
        return self.elapsed / self.n_words * 1000

    @property
    def words_per_sec(self) -> float:
        return self.n_words / self.elapsed


def make_misspelling_batch(
    sentences: Iterable[Sequence[str]],
    vocabulary: frozenset,
    n: int = 1000,
    seed: int = 99,
    min_word_length: int = 3,
) -> list[str]:
    """A batch of exactly `n` distinct non-word misspellings."""
    rng = random.Random(seed)
    words = [w for sent in sentences for w in normalise_sentence(sent)
             if w != NUM and len(w) >= min_word_length and w in vocabulary]
    rng.shuffle(words)

    batch: list[str] = []
    for word in words:
        bad = corrupt(word, vocabulary, rng, want_real_word=False)
        if bad:
            batch.append(bad)
        if len(batch) == n:
            break
    if len(batch) < n:
        raise ValueError(f"only produced {len(batch)} misspellings, needed {n}")
    return batch


def time_generator(label: str, generator: Callable[[str], set], batch: Sequence[str]) -> BenchmarkResult:
    """Push the whole batch through one candidate generator and time it."""
    total = 0
    t0 = time.perf_counter()
    for word in batch:
        total += len(generator(word))
    elapsed = time.perf_counter() - t0
    return BenchmarkResult(label, len(batch), elapsed, total)


def speed_demon(
    model: LanguageModel,
    batch: Sequence[str],
    build_times: dict[str, float] | None = None,
) -> list[BenchmarkResult]:
    """Run the identical batch through Method A and Method B."""
    gen_a = EditDistance1Generator(model.vocabulary)
    gen_b_verified = SymmetricDeleteIndex(model.vocabulary, verify=True)
    gen_b_raw = SymmetricDeleteIndex(model.vocabulary, verify=False)

    return [
        time_generator("Method A (edit-distance-1)", gen_a.candidates, batch),
        time_generator("Method B (SymSpell, verified)", gen_b_verified.candidates, batch),
        time_generator("Method B (SymSpell, raw buckets)", gen_b_raw.candidates, batch),
    ]


def error_breakdown(
    corrector: SpellingCorrector,
    cases: Sequence[TestCase],
    check_real_words: bool,
    method: str | None = None,
    top: int = 15,
) -> Counter:
    """Most common (gold -> predicted) confusions on the corrupted position."""
    conf: Counter = Counter()
    for case in cases:
        out, _ = corrector.correct_tokens(
            case.noisy, method=method, check_real_words=check_real_words
        )
        got = out[case.index]
        if got != case.original:
            conf[(case.original, case.corrupted, got)] += 1
    return Counter(dict(conf.most_common(top)))


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
