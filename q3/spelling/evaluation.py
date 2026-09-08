"""Test-set construction, accuracy metrics, and Speed Demon helpers."""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from spelling.candidates import EditDistance1Generator, SymmetricDeleteIndex
from spelling.corrector import SpellingCorrector
from spelling.language_model import LanguageModel
from spelling.tokenize import NUM, normalise_sentence


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


