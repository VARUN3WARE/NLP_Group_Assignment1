"""
Live background checks: segmentation, spelling, grammar / real-word alerts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from editor.adapters import get_q1_pipeline, get_q3_models, q3_vocabulary
from editor.config import DEFAULT_CONFIG, EditorConfig
from editor.passage import PassageToken, SampledPassage


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
    """Geometric mean of bigram and trigram perplexity (trigram preferred)."""
    if not words:
        return 0.0
    pp_bi = bigram_lm.perplexity(words)
    pp_tri = trigram_lm.perplexity(words)
    return (pp_bi * pp_tri) ** 0.5


def _acceptable_split(text: str, parts: list[str], vocab) -> bool:
    """Accept a segmentation only when pieces look like real words."""
    if len(parts) < 2:
        return False
    if "".join(parts) != text.lower():
        return False
    for w in parts:
        if w not in vocab:
            return False
        if len(w) < 2 and w not in {"a", "i"}:
            return False
    # Reject letter-by-letter chops of rare names (syme → s y me).
    if len(parts) > max(2, len(text) // 4):
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

    def process_token(
        self,
        token: PassageToken,
        token_index: int,
        accumulated: list[str],
    ) -> tuple[TokenRecord, list[Alert], float]:
        """
        Run SEGMENT then SPELL checks on one arriving token.

        Returns (record, alerts, latency_ms).
        """
        t0 = time.perf_counter()
        alerts: list[Alert] = []
        text = token.text.lower()
        final_words = [text]
        pos_tags: list[str] = []
        segment_fixed = False
        spell_fixed = False

        oov = text not in self.vocab
        long = len(text) >= self.config.long_token_chars

        # Assignment: only attempt segmentation when OOV or unusually long.
        if oov or long:
            should, pairs = self.pipe.should_split(text)
            parts = [w for w, _ in pairs] if pairs else []
            if should and _acceptable_split(text, parts, self.vocab):
                final_words = parts
                pos_tags = [t for _, t in pairs]
                segment_fixed = True
                alerts.append(
                    Alert(
                        kind="SEGMENT-ALERT",
                        message=f"split '{text}' -> {final_words} ({pos_tags})",
                        token_index=token_index,
                        details={"pairs": pairs},
                    )
                )
            elif not oov:
                pos_tags = [t for _, t in self.pipe.tag([text])]
            elif pairs and len(pairs) == 1:
                pos_tags = [t for _, t in pairs]

        # SPELL for any remaining OOV atom
        fixed_words = []
        fixed_tags = list(pos_tags) if pos_tags else []
        for i, w in enumerate(final_words):
            if w not in self.vocab:
                suggestion = self.corrector.correct_non_word(w, method="B")
                if suggestion != w:
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
                    if i < len(fixed_tags):
                        # retag after spelling change
                        pass
                    continue
            fixed_words.append(w)
        final_words = fixed_words

        if not pos_tags or spell_fixed:
            tagged = self.pipe.tag(final_words)
            pos_tags = [t for _, t in tagged]

        latency_ms = (time.perf_counter() - t0) * 1000
        record = TokenRecord(
            original=text,
            final_words=final_words,
            pos_tags=pos_tags,
            sentence_index=token.sentence_index,
            was_merge=token.is_merge,
            segment_fixed=segment_fixed,
            spell_fixed=spell_fixed,
        )
        return record, alerts, latency_ms

    def trigger_grammar_check(
        self,
        accumulated: list[str],
        token_index: int,
    ) -> tuple[list[Alert], float]:
        """Perplexity + real-word check over the last N words."""
        t0 = time.perf_counter()
        alerts: list[Alert] = []
        n = self.config.grammar_trigger_n
        window = accumulated[-n:]
        if len(window) < 2:
            return alerts, (time.perf_counter() - t0) * 1000

        if self.bigram_lm is not None and self.trigram_lm is not None:
            pp = _window_perplexity(window, self.bigram_lm, self.trigram_lm)
            if pp > self.config.grammar_perplexity_threshold:
                alerts.append(
                    Alert(
                        kind="GRAMMAR-ALERT",
                        message=(
                            f"window perplexity {pp:.1f} > "
                            f"{self.config.grammar_perplexity_threshold} :: {' '.join(window)}"
                        ),
                        token_index=token_index,
                        details={"perplexity": pp, "window": window},
                    )
                )

        # Real-word errors (Q3) on the last content word in the window
        if len(window) >= 2:
            from spelling.tokenize import BOS, EOS

            idx = len(window) - 1
            word = window[idx]
            prev = window[idx - 1] if idx > 0 else BOS
            nxt = EOS
            fixed, gain = self.corrector.correct_real_word(
                word, prev=prev, nxt=nxt, method="B"
            )
            if fixed != word and gain >= self.config.real_word_threshold:
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
                # apply correction in accumulated stream
                accumulated[-1] = fixed

        latency_ms = (time.perf_counter() - t0) * 1000
        return alerts, latency_ms

    def run_passage(self, passage: SampledPassage) -> LiveCheckResult:
        records: list[TokenRecord] = []
        alerts: list[Alert] = []
        accumulated: list[str] = []
        token_latencies: list[float] = []
        trigger_latencies: list[float] = []

        for i, tok in enumerate(passage.tokens):
            record, token_alerts, lat = self.process_token(tok, i, accumulated)
            records.append(record)
            alerts.extend(token_alerts)
            token_latencies.append(lat)
            accumulated.extend(record.final_words)

            if len(accumulated) > 0 and len(accumulated) % self.config.grammar_trigger_n == 0:
                g_alerts, g_lat = self.trigger_grammar_check(accumulated, i)
                alerts.extend(g_alerts)
                trigger_latencies.append(g_lat)

        # final trigger if leftover
        if accumulated and len(accumulated) % self.config.grammar_trigger_n != 0:
            g_alerts, g_lat = self.trigger_grammar_check(accumulated, len(passage.tokens) - 1)
            alerts.extend(g_alerts)
            trigger_latencies.append(g_lat)

        corrected_words = [w for r in records for w in r.final_words]
        corrected_tags = [t for r in records for t in r.pos_tags]

        return LiveCheckResult(
            records=records,
            alerts=alerts,
            avg_token_latency_ms=(
                sum(token_latencies) / len(token_latencies) if token_latencies else 0.0
            ),
            avg_trigger_latency_ms=(
                sum(trigger_latencies) / len(trigger_latencies) if trigger_latencies else 0.0
            ),
            corrected_words=corrected_words,
            corrected_tags=corrected_tags,
        )
