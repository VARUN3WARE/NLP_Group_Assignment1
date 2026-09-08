"""End-of-passage scoring, decision rule, and summary table."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from editor.config import DEFAULT_CONFIG, EditorConfig
from editor.live_checker import LiveCheckResult, TokenRecord
from editor.ngram_lm import AddKNgramLM
from editor.passage import SampledPassage
from editor.pcfg import PCFGParser, ParseResult
from editor.tagset import map_tagged_sentence


@dataclass
class SentenceAnalysis:
    text: str
    pcfg_result: str
    pcfg_log_prob: float | None
    bigram_log_prob: float
    trigram_log_prob: float
    chosen_method: str
    verdict: str
    segmentation_merges: int
    spelling_corrections: int


def _group_records_by_sentence(
    passage: SampledPassage, result: LiveCheckResult
) -> list[tuple[list[TokenRecord], int, int]]:
    """Return per-sentence (records, merge_count, spell_count)."""
    n = len(passage.sentences)
    buckets: list[list[TokenRecord]] = [[] for _ in range(n)]
    for rec in result.records:
        buckets[rec.sentence_index].append(rec)

    out = []
    for si, recs in enumerate(buckets):
        merges = sum(1 for r in recs if r.was_merge and r.segment_fixed)
        # count merges that existed even if not fixed, for the required column
        merges_total = sum(1 for r in recs if r.was_merge)
        spells = sum(1 for r in recs if r.spell_fixed)
        out.append((recs, merges_total, spells))
    return out


def decide_method(
    parse: ParseResult,
    bi_lp: float,
    tri_lp: float,
    all_pcfg_lps: list[float],
    config: EditorConfig = DEFAULT_CONFIG,
) -> tuple[str, str]:
    """
    Prefer PCFG when it parses and isn't a probability outlier;
    else trigram; else bigram. Return (method, verdict).
    """
    if parse.ok and parse.log_prob is not None and math.isfinite(parse.log_prob):
        if all_pcfg_lps:
            mean = sum(all_pcfg_lps) / len(all_pcfg_lps)
            var = sum((x - mean) ** 2 for x in all_pcfg_lps) / max(len(all_pcfg_lps), 1)
            std = math.sqrt(var) if var > 0 else 0.0
            if std == 0 or (mean - parse.log_prob) <= config.pcfg_outlier_z * std:
                return "pcfg", "grammatical"
            # outlier → fall through
        else:
            return "pcfg", "grammatical"

    # coverage heuristic: finite trigram score with reasonable length
    if math.isfinite(tri_lp):
        return "trigram", "plausible" if tri_lp > -80 else "implausible"
    return "bigram", "plausible" if bi_lp > -80 else "implausible"


def analyse_passage(
    passage: SampledPassage,
    live: LiveCheckResult,
    bigram_lm: AddKNgramLM,
    trigram_lm: AddKNgramLM,
    pcfg: PCFGParser,
    config: EditorConfig = DEFAULT_CONFIG,
) -> list[SentenceAnalysis]:
    groups = _group_records_by_sentence(passage, live)

    # First pass: collect PCFG log-probs for outlier detection
    prelim: list[tuple[list[str], list[str], ParseResult, float, float, int, int]] = []
    pcfg_lps: list[float] = []

    for recs, merges, spells in groups:
        words = [w for r in recs for w in r.final_words]
        tags = [t for r in recs for t in r.pos_tags]
        if len(tags) != len(words):
            tags = [t for _, t in map_tagged_sentence(list(zip(words, ["NN"] * len(words))))]
        pairs = list(zip(words, tags)) if tags else [(w, "NN") for w in words]
        parse = pcfg.parse_tagged_brown(pairs)
        bi_lp = bigram_lm.sentence_log_prob(words)
        tri_lp = trigram_lm.sentence_log_prob(words)
        if parse.ok and parse.log_prob is not None and math.isfinite(parse.log_prob):
            pcfg_lps.append(parse.log_prob)
        prelim.append((words, tags, parse, bi_lp, tri_lp, merges, spells))

    rows: list[SentenceAnalysis] = []
    for words, tags, parse, bi_lp, tri_lp, merges, spells in prelim:
        method, verdict = decide_method(parse, bi_lp, tri_lp, pcfg_lps, config)
        pcfg_cell = (
            f"logP={parse.log_prob:.2f}"
            if parse.ok and parse.log_prob is not None
            else "unparseable"
        )
        if parse.ok and parse.status == "ok-projected":
            pcfg_cell += "*"
        rows.append(
            SentenceAnalysis(
                text=" ".join(words),
                pcfg_result=pcfg_cell,
                pcfg_log_prob=parse.log_prob if parse.ok else None,
                bigram_log_prob=bi_lp,
                trigram_log_prob=tri_lp,
                chosen_method=method,
                verdict=verdict,
                segmentation_merges=merges,
                spelling_corrections=spells,
            )
        )
    return rows


def format_table(rows: list[SentenceAnalysis]) -> str:
    lines = []
    header = (
        f"{'#':<3} {'method':<8} {'verdict':<12} {'seg':>3} {'sp':>3} "
        f"{'bi_lp':>9} {'tri_lp':>9} {'pcfg':<16} sentence"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i:<3} {r.chosen_method:<8} {r.verdict:<12} {r.segmentation_merges:>3} "
            f"{r.spelling_corrections:>3} {r.bigram_log_prob:>9.1f} {r.trigram_log_prob:>9.1f} "
            f"{r.pcfg_result:<16} {r.text}"
        )
    return "\n".join(lines)
