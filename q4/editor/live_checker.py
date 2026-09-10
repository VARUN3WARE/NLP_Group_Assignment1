"""
Live background checks: segmentation, spelling, grammar / real-word alerts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from editor.adapters import get_q1_pipeline, get_q3_models, q3_vocabulary
from editor.config import DEFAULT_CONFIG, EditorConfig
from editor.passage import PassageToken, SampledPassage

# Short function words allowed as split pieces.
_SHORT_OK = frozenset(
    "a i to of in on at as is be he me my we us or an so no do go if it up by".split()
)


@dataclass
class Alert:
    kind: str  # SEGMENT-ALERT | SPELL-ALERT | GRAMMAR-ALERT
    message: str
    token_index: int
    details: dict = field(default_factory=dict)


@dataclass
class TokenRecord:
    original: str
    final_words: list[str]
    pos_tags: list[str]
    sentence_index: int
    was_merge: bool
    segment_fixed: bool = False
    spell_fixed: bool = False


@dataclass
class LiveCheckResult:
    records: list[TokenRecord]
    alerts: list[Alert]
    avg_token_latency_ms: float
    avg_trigger_latency_ms: float
    corrected_words: list[str]
    corrected_tags: list[str]


def _window_perplexity(words: list[str], bigram_lm, trigram_lm) -> float:
    if not words:
        return 0.0
    pp_bi = bigram_lm.perplexity(words)
    pp_tri = trigram_lm.perplexity(words)
    return (pp_bi * pp_tri) ** 0.5


def _acceptable_split(text: str, parts: list[str], vocab) -> bool:
    """
    Accept segmentation only for plausible spacebar-miss merges.

    Rejects chops like jackal→jack+al, aha→a+ha, saumur→letters.
    Prefers exactly two solid pieces whose lengths look like two words.
    """
    text = text.lower()
    if len(parts) < 2:
        return False
    if "".join(parts) != text:
        return False

    # Prefer binary merges (the simulated error type).
    if len(parts) > 2:
        return False

    for w in parts:
        if w not in vocab:
            return False
        if len(w) < 2 and w not in _SHORT_OK:
            return False
        # Reject tiny junk pieces like "al", "ha" unless they are allow-listed.
        if len(w) == 2 and w not in _SHORT_OK:
            return False

    left, right = parts[0], parts[1]
    # Both sides should carry real mass for longer tokens.
    if len(text) >= 8 and min(len(left), len(right)) < 3:
        if left not in _SHORT_OK and right not in _SHORT_OK:
            return False

    # Reject almost-even letter chops of a single morphological word
    # when one side is a proper-name fragment pattern (very short + short).
    if len(left) <= 2 and len(right) <= 2 and text not in vocab:
        return False

    return True


def _best_binary_split(text: str, vocab) -> list[str] | None:
    """
    Fallback when the joint decoder chops into >2 pieces (e.g. saidthank).
    Prefer a single cut into two in-vocab words (spacebar-miss model).
    """
    text = text.lower()
    if len(text) < 4:
        return None
    best: tuple[float, list[str]] | None = None
    for i in range(1, len(text)):
        left, right = text[:i], text[i:]
        if not _acceptable_split(text, [left, right], vocab):
            continue
        # Prefer longer min-side (said|thank over sai|dthank-style if any).
        score = min(len(left), len(right)) + 0.01 * max(len(left), len(right))
        if best is None or score > best[0]:
            best = (score, [left, right])
    return best[1] if best else None


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _robust_z_score(pp: float, history: list[float]) -> float:
    """
    Robust z-score using median + MAD (median absolute deviation).
    Returns how many 'sigma' above the baseline the value is.
    """
    if len(history) < 2:
        return 0.0
    med = _median(history)
    mad = _median([abs(v - med) for v in history])
    # 1.4826 makes MAD comparable to std-dev for normal distributions.
    sigma = mad * 1.4826
    if sigma < 1e-6:
        return 0.0
    return (pp - med) / sigma


_SUFFIXES = ("er", "ing", "ed", "ly", "est", "ers", "est")


def _ends_in_suffix(word: str) -> str | None:
    """Return the matched English suffix, or None."""
    for suf in _SUFFIXES:
        if len(word) > len(suf) + 2 and word.endswith(suf):
            return suf
    return None


def _safe_spell_suggestion(original: str, suggestion: str, vocab) -> bool:
    """Filter clearly harmful unigram spell replacements."""
    if not suggestion or suggestion == original:
        return False
    if suggestion not in vocab:
        return False
    # Never auto-fix very short tokens (mr→or, ha→a, trills→trials, sluing→slung).
    # Require length >= 6 so rare but valid literary words (5-6 chars) are left alone.
    if len(original) < 6:
        return False
    # Keep length close (edit-distance-1 already, but guard insertions/deletions).
    if abs(len(suggestion) - len(original)) > 1:
        return False
    # Prefer keeping first character for nouns/content words (crab≠grab).
    if original[0] != suggestion[0]:
        return False
    # Same-length swaps need a real shared stem (trills≠trials: prefix "tri"=3).
    if len(original) == len(suggestion) and _common_prefix_len(original, suggestion) < 4:
        return False
    # Protect words ending in a common English suffix (caresser→caresses, sluing→slung).
    orig_suf = _ends_in_suffix(original)
    if orig_suf is not None and not suggestion.endswith(orig_suf):
        return False
    return True


class LiveChecker:
    def __init__(
        self,
        config: EditorConfig = DEFAULT_CONFIG,
        bigram_lm=None,
        trigram_lm=None,
    ):
        self.config = config
        self.pipe = get_q1_pipeline()
        self.spell_model, self.corrector = get_q3_models(
            method="B", real_word_threshold=config.real_word_threshold
        )
        self.vocab = q3_vocabulary()
        self.bigram_lm = bigram_lm
        self.trigram_lm = trigram_lm
        # Running window perplexity history for adaptive (relative) grammar alerts.
        self._pp_history: list[float] = []

    def process_token(
        self,
        token: PassageToken,
        token_index: int,
        accumulated: list[str],
    ) -> tuple[TokenRecord, list[Alert], float]:
        t0 = time.perf_counter()
        alerts: list[Alert] = []
        text = token.text.lower()
        # Strip trailing punctuation for checks; keep alphanumeric+' tokens only
        text = "".join(ch for ch in text if ch.isalnum() or ch == "'")
        if not text:
            return (
                TokenRecord(token.text, [token.text], [], token.sentence_index, token.is_merge),
                [],
                0.0,
            )

        final_words = [text]
        pos_tags: list[str] = []
        segment_fixed = False
        spell_fixed = False

        oov = text not in self.vocab
        long = len(text) >= self.config.long_token_chars

        if oov or long:
            should, pairs = self.pipe.should_split(text)
            parts = [w for w, _ in pairs] if pairs else []
            if should and _acceptable_split(text, parts, self.vocab):
                final_words = parts
                pos_tags = [t for _, t in pairs]
                segment_fixed = True
            else:
                # Q1 sometimes over-segments (saidthank→said+than+k); try binary cut.
                binary = _best_binary_split(text, self.vocab)
                if binary is not None:
                    final_words = binary
                    tagged = self.pipe.tag(final_words)
                    pos_tags = [t for _, t in tagged]
                    pairs = list(zip(final_words, pos_tags))
                    segment_fixed = True
                elif not oov:
                    pos_tags = [t for _, t in self.pipe.tag([text])]
                elif pairs and len(pairs) == 1:
                    pos_tags = [t for _, t in pairs]

            if segment_fixed:
                alerts.append(
                    Alert(
                        kind="SEGMENT-ALERT",
                        message=f"split '{text}' -> {final_words} ({pos_tags})",
                        token_index=token_index,
                        details={"pairs": pairs},
                    )
                )

        fixed_words = []
        for w in final_words:
            if w not in self.vocab:
                suggestion = self.corrector.correct_non_word(w, method="B")
                if _safe_spell_suggestion(w, suggestion, self.vocab):
                    spell_fixed = True
                    alerts.append(
                        Alert(
                            kind="SPELL-ALERT",
                            message=f"'{w}' -> '{suggestion}' (Method B / unigram)",
                            token_index=token_index,
                            details={"from": w, "to": suggestion},
                        )
                    )
                    fixed_words.append(suggestion)
                    continue
            fixed_words.append(w)
        final_words = fixed_words

        if not pos_tags or spell_fixed or len(pos_tags) != len(final_words):
            tagged = self.pipe.tag(final_words)
            pos_tags = [t for _, t in tagged]

        latency_ms = (time.perf_counter() - t0) * 1000
        return (
            TokenRecord(
                original=text,
                final_words=final_words,
                pos_tags=pos_tags,
                sentence_index=token.sentence_index,
                was_merge=token.is_merge,
                segment_fixed=segment_fixed,
                spell_fixed=spell_fixed,
            ),
            alerts,
            latency_ms,
        )

    def trigger_grammar_check(
        self,
        accumulated: list[str],
        token_index: int,
        *,
        apply_real_word: bool = True,
    ) -> tuple[list[Alert], float]:
        t0 = time.perf_counter()
        alerts: list[Alert] = []
        n = self.config.grammar_trigger_n
        window = accumulated[-n:]
        if len(window) < 2:
            return alerts, (time.perf_counter() - t0) * 1000

        if self.bigram_lm is not None and self.trigram_lm is not None:
            pp = _window_perplexity(window, self.bigram_lm, self.trigram_lm)
            # Adaptive relative outlier detection: alert only when this window's
            # perplexity is a robust z-score outlier above the passage's own
            # running baseline (median + MAD).  This avoids the constant noise
            # of an absolute threshold when Brown-trained LMs meet Gutenberg text.
            z = _robust_z_score(pp, self._pp_history)
            warmup = self.config.grammar_warmup_windows
            should_alert = (
                len(self._pp_history) >= warmup
                and z >= self.config.grammar_z_threshold
            )
            # Always record this window's PP for the running baseline.
            self._pp_history.append(pp)
            if should_alert:
                alerts.append(
                    Alert(
                        kind="GRAMMAR-ALERT",
                        message=(
                            f"window perplexity {pp:.1f} "
                            f"(z={z:.1f} > {self.config.grammar_z_threshold}) "
                            f":: {' '.join(window)}"
                        ),
                        token_index=token_index,
                        details={
                            "perplexity": pp,
                            "z_score": z,
                            "window": list(window),
                        },
                    )
                )

        if apply_real_word and len(window) >= 2:
            from spelling.tokenize import BOS, EOS

            word = window[-1]
            # Skip real-word checks on tiny / closed-class tokens.
            if len(word) >= 4 and word not in _SHORT_OK:
                prev = window[-2] if len(window) >= 2 else BOS
                fixed, gain = self.corrector.correct_real_word(
                    word, prev=prev, nxt=EOS, method="B"
                )
                # Live editor uses a stricter margin than Q3's default (5.0) —
                # see REPORT.md "live real-word margin" (≥18 nats).
                margin = max(self.config.real_word_threshold, 18.0)
                if fixed != word and gain >= margin and fixed[0] == word[0]:
                    alerts.append(
                        Alert(
                            kind="GRAMMAR-ALERT",
                            message=(
                                f"real-word '{word}' -> '{fixed}' "
                                f"(+{gain:.2f} nats) in context"
                            ),
                            token_index=token_index,
                            details={
                                "from": word,
                                "to": fixed,
                                "gain": gain,
                                "kind": "real-word",
                            },
                        )
                    )
                    accumulated[-1] = fixed

        return alerts, (time.perf_counter() - t0) * 1000

    def reset_grammar_stats(self) -> None:
        """Clear the running perplexity baseline (call between passages)."""
        self._pp_history.clear()

    def run_passage(self, passage: SampledPassage) -> LiveCheckResult:
        records: list[TokenRecord] = []
        alerts: list[Alert] = []
        accumulated: list[str] = []
        token_latencies: list[float] = []
        trigger_latencies: list[float] = []
        self.reset_grammar_stats()

        for i, tok in enumerate(passage.tokens):
            record, token_alerts, lat = self.process_token(tok, i, accumulated)
            records.append(record)
            alerts.extend(token_alerts)
            token_latencies.append(lat)
            accumulated.extend(record.final_words)

            if accumulated and len(accumulated) % self.config.grammar_trigger_n == 0:
                g_alerts, g_lat = self.trigger_grammar_check(accumulated, i)
                alerts.extend(g_alerts)
                trigger_latencies.append(g_lat)

        if accumulated and len(accumulated) % self.config.grammar_trigger_n != 0:
            g_alerts, g_lat = self.trigger_grammar_check(
                accumulated, len(passage.tokens) - 1
            )
            alerts.extend(g_alerts)
            trigger_latencies.append(g_lat)

        return LiveCheckResult(
            records=records,
            alerts=alerts,
            avg_token_latency_ms=(
                sum(token_latencies) / len(token_latencies) if token_latencies else 0.0
            ),
            avg_trigger_latency_ms=(
                sum(trigger_latencies) / len(trigger_latencies) if trigger_latencies else 0.0
            ),
            corrected_words=[w for r in records for w in r.final_words],
            corrected_tags=[t for r in records for t in r.pos_tags],
        )
