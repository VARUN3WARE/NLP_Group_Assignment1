#!/usr/bin/env python3
"""
Question 1 CLI.

    python main.py train-english   train + pickle English models for Q4
    python main.py sample          demo sample strings (loads or trains English)
    python main.py evaluate        full evaluation via results.py (slow)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pipeline import DEFAULT_BUNDLE_PATH, load_english_pipeline, train_english_pipeline


def cmd_train_english(args: argparse.Namespace) -> None:
    print("Training English trigram LM + POS tagger on Brown (80% train)...")
    t0 = time.perf_counter()
    pipe = train_english_pipeline(
        max_word_length=args.max_word_length,
        alpha=args.alpha,
        beta=args.beta,
        beam_width=args.beam_width,
    )
    train_s = time.perf_counter() - t0
    path = pipe.save(args.out)
    print(f"  {pipe.lm}")
    print(f"  tags={len(pipe.tagger.tags):,}  vocab={len(pipe.tagger.vocabulary):,}")
    print(f"  trained in {train_s:.1f}s")
    print(f"  saved -> {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    print(
        f"  decoder: max_word_length={pipe.max_word_length} "
        f"α={pipe.alpha} β={pipe.beta} beam={pipe.beam_width}"
    )


def _get_pipeline(args: argparse.Namespace):
    path = Path(args.model)
    if path.exists():
        print(f"Loading English pipeline from {path} ...")
        return load_english_pipeline(path)
    print("No saved model found; training English pipeline once ...")
    pipe = train_english_pipeline(
        max_word_length=args.max_word_length,
        alpha=args.alpha,
        beta=args.beta,
        beam_width=args.beam_width,
    )
    pipe.save(path)
    return pipe


def cmd_sample(args: argparse.Namespace) -> None:
    pipe = _get_pipeline(args)

    english = [
        "thequickbrownfoxjumpsoverthelazydog",
        "thequickbrownfox",
        "tobeornottobe",
    ]
    print("\nEnglish joint beam decode")
    for text in english:
        pairs = pipe.decode(text)
        print(f"  {text}")
        print(f"  -> {pairs}\n")

    # Spanish samples still use the live modules (no Spanish pickle required for Q4).
    try:
        from corpus import load_spanish
        from language_model import TrigramLanguageModel
        from pos_tagger import MorphologyAwarePOSTagger, TrigramPOSTagger
        from segmentation import ViterbiSegmenter

        print("Loading Spanish-GSD for sample demos ...")
        train, _dev, _test = load_spanish()
        words = [[t["form"].lower() for t in sent] for sent in train]
        lm = TrigramLanguageModel()
        lm.train(words)
        seg = ViterbiSegmenter(lm)
        plain = TrigramPOSTagger()
        plain.train([[(t["form"], t["upos"]) for t in sent] for sent in train])
        morph = MorphologyAwarePOSTagger()
        morph.train_spanish(train)

        spanish = ["lacasarojaesgrande", "mispadrespuedenviajar", "elcielodespejadoesazul"]
        print("\nSpanish segment + plain POS / morph POS")
        for text in spanish:
            w = seg.segment(text)
            print(f"  {text}")
            print(f"  plain -> {plain.tag(w)}")
            print(f"  morph -> {morph.tag(w)}\n")
    except FileNotFoundError as exc:
        print(f"\nSkipping Spanish samples ({exc})")


def cmd_evaluate(_args: argparse.Namespace) -> None:
    print("Running full evaluation (results.py) ...")
    import results  # noqa: F401 — script runs on import

    # results.py executes at import time; keep an explicit no-op for clarity.
    _ = results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_decoder_opts(p):
        p.add_argument("--max-word-length", type=int, default=20)
        p.add_argument("--alpha", type=float, default=1.0, help="LM weight in joint score")
        p.add_argument("--beta", type=float, default=1.0, help="POS weight in joint score")
        p.add_argument("--beam-width", type=int, default=8)

    p = sub.add_parser("train-english", help="Train and pickle English models for Q4")
    p.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_PATH)
    add_decoder_opts(p)
    p.set_defaults(func=cmd_train_english)

    p = sub.add_parser("sample", help="Print sample segment/POS outputs")
    p.add_argument("--model", type=Path, default=DEFAULT_BUNDLE_PATH)
    add_decoder_opts(p)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("evaluate", help="Run full metrics (slow)")
    p.set_defaults(func=cmd_evaluate)

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
