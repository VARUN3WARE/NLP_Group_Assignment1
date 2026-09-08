#!/usr/bin/env python3
"""
Question 4 CLI.

    python main.py train-lms     train Q4 bigram/trigram (+ PCFG cache)
    python main.py simulate      sample passage, stream alerts, print table
    python main.py analyze       same as simulate (alias) with saved results
    python main.py bench         1,000-word Speed Demon
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make q4/ importable as root for `editor.*`
Q4_ROOT = Path(__file__).resolve().parent
if str(Q4_ROOT) not in sys.path:
    sys.path.insert(0, str(Q4_ROOT))

from editor.analysis import analyse_passage, format_table
from editor.benchmark import run_speed_demon
from editor.config import DEFAULT_CONFIG, EditorConfig
from editor.live_checker import LiveChecker
from editor.ngram_lm import load_q4_language_models, train_q4_language_models
from editor.passage import sample_passage
from editor.paths import RESULTS_DIR
from editor.pcfg import load_or_train_pcfg


def cmd_train_lms(args: argparse.Namespace) -> None:
    print("Training Q4 add-k bigram + trigram on Brown ...")
    bi, tri = train_q4_language_models(k=args.k)
    print(" ", bi)
    print(" ", tri)
    print("Training / caching PCFG from Penn Treebank ...")
    pcfg = load_or_train_pcfg(max_trees=args.max_trees)
    print(f"  grammar productions: {len(pcfg.grammar.productions())}")


def _run_pipeline(args: argparse.Namespace, save: bool) -> None:
    cfg = EditorConfig(
        merge_probability=args.p,
        grammar_trigger_n=args.n,
        add_k=args.k,
        typing_delay_s=args.delay,
    )
    print("Loading Q4 LMs + PCFG (and Q1/Q3 adapters on first use) ...")
    bi, tri = load_q4_language_models(k=cfg.add_k)
    pcfg = load_or_train_pcfg()
    checker = LiveChecker(config=cfg, bigram_lm=bi, trigram_lm=tri)

    passage = sample_passage(config=cfg, seed=args.seed)
    print(f"\nPassage from {passage.file_id}  ({len(passage.sentences)} sentences, "
          f"{len(passage.tokens)} tokens, p={cfg.merge_probability})")
    print("Gold:", passage.gold_text[:200], "...")
    print("Stream (with merges):", passage.plain_text[:200], "...\n")

    print("=== LIVE ALERTS ===")
    # Stream word-by-word for demo
    from editor.live_checker import LiveCheckResult, TokenRecord, Alert

    records = []
    alerts = []
    accumulated = []
    token_lats = []
    trig_lats = []

    for i, tok in enumerate(passage.tokens):
        if args.delay > 0:
            time.sleep(args.delay)
        rec, al, lat = checker.process_token(tok, i, accumulated)
        records.append(rec)
        alerts.extend(al)
        token_lats.append(lat)
        accumulated.extend(rec.final_words)
        for a in al:
            print(f"  [{a.kind}] {a.message}")

        if len(accumulated) % cfg.grammar_trigger_n == 0:
            g_al, g_lat = checker.trigger_grammar_check(accumulated, i)
            alerts.extend(g_al)
            trig_lats.append(g_lat)
            for a in g_al:
                print(f"  [{a.kind}] {a.message}")

    if accumulated and len(accumulated) % cfg.grammar_trigger_n != 0:
        g_al, g_lat = checker.trigger_grammar_check(accumulated, len(passage.tokens) - 1)
        alerts.extend(g_al)
        trig_lats.append(g_lat)
        for a in g_al:
            print(f"  [{a.kind}] {a.message}")

    live = LiveCheckResult(
        records=records,
        alerts=alerts,
        avg_token_latency_ms=sum(token_lats) / len(token_lats) if token_lats else 0.0,
        avg_trigger_latency_ms=sum(trig_lats) / len(trig_lats) if trig_lats else 0.0,
        corrected_words=[w for r in records for w in r.final_words],
        corrected_tags=[t for r in records for t in r.pos_tags],
    )

    print(f"\nLatency: seg+spell {live.avg_token_latency_ms:.2f} ms/token | "
          f"grammar trigger {live.avg_trigger_latency_ms:.2f} ms/trigger")

    rows = analyse_passage(passage, live, bi, tri, pcfg, cfg)
    print("\n=== END-OF-PASSAGE TABLE ===")
    print(format_table(rows))

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "file_id": passage.file_id,
            "seed": passage.seed,
            "p": cfg.merge_probability,
            "N": cfg.grammar_trigger_n,
            "avg_token_latency_ms": live.avg_token_latency_ms,
            "avg_trigger_latency_ms": live.avg_trigger_latency_ms,
            "n_alerts": len(alerts),
            "alert_kinds": {
                k: sum(1 for a in alerts if a.kind == k)
                for k in ("SEGMENT-ALERT", "SPELL-ALERT", "GRAMMAR-ALERT")
            },
            "sentences": [
                {
                    "text": r.text,
                    "pcfg": r.pcfg_result,
                    "bigram_lp": r.bigram_log_prob,
                    "trigram_lp": r.trigram_log_prob,
                    "method": r.chosen_method,
                    "verdict": r.verdict,
                    "seg_merges": r.segmentation_merges,
                    "spell_fixes": r.spelling_corrections,
                }
                for r in rows
            ],
        }
        out = RESULTS_DIR / f"run_{passage.file_id.replace('.','-')}_{int(time.time())}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved -> {out}")


def cmd_simulate(args: argparse.Namespace) -> None:
    _run_pipeline(args, save=False)


def cmd_analyze(args: argparse.Namespace) -> None:
    _run_pipeline(args, save=True)


def cmd_bench(args: argparse.Namespace) -> None:
    print(f"Speed Demon: {args.words} words ...")
    result = run_speed_demon(n=args.words)
    print(f"  seg+spell: {result.seg_spell_total_s:.3f}s  ({result.seg_spell_per_word_ms:.3f} ms/word)")
    print(f"  grammar:   {result.grammar_total_s:.3f}s  ({result.grammar_per_word_ms:.3f} ms/word)")
    print(f"  overhead:  {result.seg_spell_overhead_ms:.3f} ms/word")
    print(result.conclusion)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train-lms", help="Train Q4 n-gram LMs and cache PCFG")
    p.add_argument("--k", type=float, default=DEFAULT_CONFIG.add_k)
    p.add_argument("--max-trees", type=int, default=2000)
    p.set_defaults(func=cmd_train_lms)

    def add_sim_opts(p):
        p.add_argument("--p", type=float, default=DEFAULT_CONFIG.merge_probability)
        p.add_argument("--n", type=int, default=DEFAULT_CONFIG.grammar_trigger_n)
        p.add_argument("--k", type=float, default=DEFAULT_CONFIG.add_k)
        p.add_argument("--delay", type=float, default=0.0, help="sleep seconds between tokens")
        p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("simulate", help="Stream a random passage with live alerts")
    add_sim_opts(p)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("analyze", help="Simulate + save results JSON")
    add_sim_opts(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("bench", help="1,000-word Speed Demon")
    p.add_argument("--words", type=int, default=1000)
    p.set_defaults(func=cmd_bench)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
