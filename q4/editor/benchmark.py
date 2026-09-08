"""Speed Demon benchmark for Part 5."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from editor.adapters import get_q3_models
from editor.config import DEFAULT_CONFIG, EditorConfig
from editor.live_checker import LiveChecker
from editor.ngram_lm import load_q4_language_models
from editor.passage import PassageToken
from editor.paths import RESULTS_DIR


@dataclass
class SpeedDemonResult:
    n_words: int
    seg_spell_total_s: float
    seg_spell_per_word_ms: float
    grammar_total_s: float
    grammar_per_word_ms: float
    seg_spell_overhead_ms: float
    conclusion: str


def run_speed_demon(
    n: int = 1000,
    config: EditorConfig = DEFAULT_CONFIG,
    out_path: Path | None = None,
) -> SpeedDemonResult:
    """
    1,000 simulated tokens through (a) seg+spell and (b) grammar-trigger only.
    Reuses Q3 misspelling batch generator.
    """
    import sys
    from pathlib import Path as P

    repo = P(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from q1_paths import ensure_q3_on_path

    ensure_q3_on_path()
    from spelling import make_misspelling_batch
    import pickle
    from spelling.paths import DEFAULT_SPLIT_PATH

    model, _ = get_q3_models(method="B")
    with open(DEFAULT_SPLIT_PATH, "rb") as fh:
        test_sentences = pickle.load(fh)

    batch = make_misspelling_batch(test_sentences, model.vocabulary, n=n)
    assert len(batch) == n

    bi, tri = load_q4_language_models(k=config.add_k)
    checker = LiveChecker(config=config, bigram_lm=bi, trigram_lm=tri)

    # (a) full per-token seg+spell
    t0 = time.perf_counter()
    accumulated: list[str] = []
    for i, word in enumerate(batch):
        tok = PassageToken(text=word, gold_words=[word], is_merge=False, sentence_index=0)
        rec, _alerts, _lat = checker.process_token(tok, i, accumulated)
        accumulated.extend(rec.final_words)
    seg_spell_total = time.perf_counter() - t0

    # (b) grammar-trigger only on the same corrected stream, in windows of N
    # Build a clean-ish stream: use corrected words from a second pass quickly
    words = list(accumulated)
    t0 = time.perf_counter()
    window: list[str] = []
    for i, w in enumerate(words):
        window.append(w)
        if len(window) % config.grammar_trigger_n == 0:
            checker.trigger_grammar_check(window, i)
    if len(window) % config.grammar_trigger_n != 0:
        checker.trigger_grammar_check(window, len(words) - 1)
    grammar_total = time.perf_counter() - t0

    seg_ms = seg_spell_total / n * 1000
    gram_ms = grammar_total / n * 1000
    overhead = seg_ms - gram_ms

    conclusion = (
        f"Segmentation+spelling costs {seg_ms:.3f} ms/word vs "
        f"{gram_ms:.3f} ms/word for grammar-trigger checks "
        f"(overhead {overhead:.3f} ms/word). "
        f"{'Live per-token seg+spell is cheap enough to keep enabled.' if seg_ms < 5 else 'Consider throttling seg+spell to the grammar trigger interval.'}"
    )

    result = SpeedDemonResult(
        n_words=n,
        seg_spell_total_s=seg_spell_total,
        seg_spell_per_word_ms=seg_ms,
        grammar_total_s=grammar_total,
        grammar_per_word_ms=gram_ms,
        seg_spell_overhead_ms=overhead,
        conclusion=conclusion,
    )

    out_path = out_path or (RESULTS_DIR / "speed_demon.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2))
    return result
