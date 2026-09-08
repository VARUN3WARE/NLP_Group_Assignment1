#!/usr/bin/env python3
"""
Question 3 - drivers for every part of the assignment.

    python main.py train      Part 1  train and pickle the models
    python main.py eval       Part 4  accuracy on both test sets
    python main.py bench      Part 4  the 1,000-word "Speed Demon" benchmark
    python main.py demo       Part 5  output examples, non-interactively
    python main.py app        Part 5  the live interactive application
    python main.py all                train + eval + bench + demo

Every command writes its numbers into results/results.json alongside printing
them.  `python main.py <command> --help` lists that command's options.
"""

from __future__ import annotations

# Ctrl+C during start-up used to land inside whichever module happened to be
# importing and print its traceback.  Silence it before anything slow is
# imported, then hand SIGINT back to Python so the interactive loop below can
# still catch KeyboardInterrupt and print its session summary.
import signal
import sys as _sys


def _quiet_sigint(signum, frame):
    _sys.stderr.write("\n")
    raise SystemExit(130)


_default_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, _quiet_sigint)

import argparse
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

from spelling import (
    ALPHABET,
    DEFAULT_MODEL_PATH,
    DEFAULT_SPLIT_PATH,
    RESULTS_DIR,
    EditDistance1Generator,
    LanguageModel,
    SpellingCorrector,
    SymmetricDeleteIndex,
    brown_sentences,
    build_test_sets,
    error_breakdown,
    evaluate,
    false_alarm_rate,
    load_models,
    make_misspelling_batch,
    threshold_sweep,
    train_test_split,
)

signal.signal(signal.SIGINT, _default_sigint)

RESULTS_PATH = RESULTS_DIR / "results.json"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78 + "\n")


def save_results(section: str, payload: dict) -> None:
    """Merge one section into results/results.json, keeping the others."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if RESULTS_PATH.exists():
        try:
            data = json.loads(RESULTS_PATH.read_text())
        except json.JSONDecodeError:
            data = {}
    data[section] = payload
    RESULTS_PATH.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nSaved -> {RESULTS_PATH}  [{section}]")


def require_models() -> tuple[LanguageModel, list]:
    """Load the trained model and the held-out split, or explain how to get them."""
    if not DEFAULT_MODEL_PATH.exists():
        sys.exit("No trained model found. Run:  python main.py train")
    model = LanguageModel.load(DEFAULT_MODEL_PATH)
    with open(DEFAULT_SPLIT_PATH, "rb") as fh:
        test_sentences = pickle.load(fh)
    return model, test_sentences


# --------------------------------------------------------------------------- #
# Part 1 - train
# --------------------------------------------------------------------------- #

def cmd_train(args: argparse.Namespace) -> None:
    """Train the vocabulary, unigram and bigram models on the Brown corpus.

    10% of the sentences are held out *before* anything is counted, so the
    Part 4 accuracy figures are measured on unseen text.
    """
    print("Loading the Brown corpus ...")
    t0 = time.perf_counter()
    sentences = brown_sentences()
    print(f"  {len(sentences):,} sentences  ({time.perf_counter() - t0:.1f}s)")

    train, test = train_test_split(sentences, args.test_ratio, args.seed)
    print(f"  train: {len(train):,} sentences   held-out test: {len(test):,} sentences")

    print("\nTraining unigram + bigram models on the training split ...")
    t0 = time.perf_counter()
    model = LanguageModel.train(train, k=args.k)
    train_time = time.perf_counter() - t0
    print(f"  {model}   ({train_time:.1f}s)")

    print("\nBuilding the Method B symmetric-delete index (one-off cost) ...")
    t0 = time.perf_counter()
    index = SymmetricDeleteIndex(model.vocabulary)
    index_time = time.perf_counter() - t0
    print(f"  {index}   ({index_time:.1f}s)")

    model.save(DEFAULT_MODEL_PATH)
    with open(DEFAULT_SPLIT_PATH, "wb") as fh:
        pickle.dump(test, fh, protocol=pickle.HIGHEST_PROTOCOL)

    top = model.unigram_counts.most_common(10)
    print(f"\nSaved model  -> {DEFAULT_MODEL_PATH}  "
          f"({DEFAULT_MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved split  -> {DEFAULT_SPLIT_PATH}")
    print("\nMost frequent words:", ", ".join(f"{w}({c})" for w, c in top))

    save_results("model", {
        "brown_sentences": len(sentences),
        "train_sentences": len(train),
        "test_sentences": len(test),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "add_k": args.k,
        "vocabulary_types": len(model.vocabulary),
        "training_tokens": model.total_tokens,
        "distinct_bigrams": len(model.bigram_counts),
        "symspell_keys": index.n_entries,
        "symspell_postings": index.n_postings,
        "symspell_build_seconds": round(index_time, 3),
        "lm_train_seconds": round(train_time, 3),
        "model_file_mb": round(DEFAULT_MODEL_PATH.stat().st_size / 1e6, 2),
        "top_unigrams": top,
    })


# --------------------------------------------------------------------------- #
# Part 4 - accuracy
# --------------------------------------------------------------------------- #

EVAL_HEADER = (f"{'test set':<22}{'n':>7}{'target acc':>11}{'sent acc':>12}"
               f"{'no change':>11}{'wrong fix':>11}{'collateral':>12}{'time':>11}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Accuracy on the paired non-word and real-word test sets."""
    model, test_sentences = require_models()
    print(f"Model: {model}")
    print(f"Held-out sentences: {len(test_sentences):,}")

    cap = None if args.full else args.max_cases
    t0 = time.perf_counter()
    non_word, real_word = build_test_sets(
        test_sentences, model.vocabulary, seed=args.seed, max_cases=cap
    )
    print(f"Test sets built in {time.perf_counter() - t0:.1f}s: "
          f"{len(non_word):,} non-word / {len(real_word):,} real-word cases\n")

    print("Example corruptions")
    for case in non_word[:3]:
        rw = next(c for c in real_word if c.tokens is case.tokens)
        print(f"  gold '{case.original}'  ->  non-word '{case.corrupted}'  "
              f"|  real-word '{rw.corrupted}'")
    print()

    corrector = SpellingCorrector(
        model, method=args.method, real_word_threshold=args.threshold
    )
    results = {}
    print(EVAL_HEADER)
    print("-" * len(EVAL_HEADER))

    # Non-word set, real-word checking OFF: this isolates Part 3.1 exactly.
    r = evaluate(corrector, non_word, check_real_words=False, method=args.method)
    print(r.as_row("non-word only"))
    results["non_word_isolated"] = r.__dict__ | {"accuracy": r.accuracy}

    # Non-word set with the full pipeline (real-word checking can undo good work).
    r = evaluate(corrector, non_word, check_real_words=True, method=args.method)
    print(r.as_row("non-word + rw check"))
    results["non_word_full_pipeline"] = r.__dict__ | {"accuracy": r.accuracy}

    # Real-word set: this is the whole point of the bigram model.
    r = evaluate(corrector, real_word, check_real_words=True, method=args.method)
    print(r.as_row("real-word"))
    results["real_word"] = r.__dict__ | {"accuracy": r.accuracy}

    # Real-word set with context checking OFF - the "do nothing" baseline, which
    # by construction scores 0: every corrupted word is itself a legal word.
    r = evaluate(corrector, real_word, check_real_words=False, method=args.method)
    print(r.as_row("real-word (no ctx)"))
    results["real_word_no_context"] = r.__dict__ | {"accuracy": r.accuracy}

    # ---- false alarms on clean text ------------------------------------- #
    clean = real_word[:1000]
    fa = false_alarm_rate(corrector, clean, method=args.method)
    print(f"\nFalse alarms on {fa.n_sentences:,} *uncorrupted* sentences "
          f"({fa.n_words:,} words):")
    print(f"  real-word rewrites : {fa.real_word_changes:,} "
          f"({fa.rate_per_word:.2%} of words, {fa.rate_per_sentence:.2%} of sentences)")
    print(f"  OOV rewrites       : {fa.non_word_changes:,} "
          f"(words the training split never saw - not false alarms)")
    results["false_alarms"] = fa.__dict__ | {
        "rate_per_word": fa.rate_per_word,
        "rate_per_sentence": fa.rate_per_sentence,
    }

    # ---- threshold trade-off -------------------------------------------- #
    print("\nAccuracy / false-alarm trade-off (real-word set)")
    print(f"{'threshold':>10}{'rw accuracy':>14}{'FA per word':>14}{'FA per sent':>14}")
    sweep = threshold_sweep(model, real_word, clean, method="B")
    for row in sweep:
        print(f"{row['threshold']:>10}{row['real_word_accuracy']:>14.2%}"
              f"{row['false_alarm_per_word']:>14.2%}"
              f"{row['false_alarm_per_sentence']:>14.2%}")
    results["threshold_sweep"] = sweep

    # ---- add-k vs interpolated ------------------------------------------ #
    print(f"\nScoring rule comparison at a matched threshold "
          f"(threshold={args.threshold})")
    print(f"{'scoring':>16}{'beta':>7}{'rw accuracy':>14}{'FA per word':>14}")
    comparison = []
    for scoring, beta in (("add-k", 0.0), ("interpolated", 0.0),
                          ("interpolated", 0.5), ("interpolated", 1.0)):
        c = SpellingCorrector(model, method="B", real_word_threshold=args.threshold,
                              scoring=scoring, unigram_discount=beta)
        acc = evaluate(c, real_word, check_real_words=True)
        f = false_alarm_rate(c, clean)
        print(f"{scoring:>16}{beta:>7}{acc.accuracy:>14.2%}{f.rate_per_word:>14.2%}")
        comparison.append({"scoring": scoring, "unigram_discount": beta,
                           "real_word_accuracy": acc.accuracy,
                           "false_alarm_per_word": f.rate_per_word})
    results["scoring_comparison"] = comparison

    # ---- residual errors -------------------------------------------------#
    print("\nMost common residual non-word errors  (gold | typed | predicted)")
    for (gold, typed, got), n in error_breakdown(
            corrector, non_word, check_real_words=False, top=12).most_common():
        print(f"  {n:>3}  {gold:<14} {typed:<14} -> {got}")

    print("\nMost common residual real-word errors  (gold | typed | predicted)")
    for (gold, typed, got), n in error_breakdown(
            corrector, real_word, check_real_words=True, top=12).most_common():
        print(f"  {n:>3}  {gold:<14} {typed:<14} -> {got}")

    results["config"] = {
        "threshold": args.threshold, "method": args.method,
        "n_non_word": len(non_word), "n_real_word": len(real_word),
        "seed": args.seed, "full": args.full,
    }
    save_results("evaluation", results)


# --------------------------------------------------------------------------- #
# Part 4 - the "Speed Demon" benchmark
# --------------------------------------------------------------------------- #

def strings_built_method_a(word: str, alphabet_size: int) -> int:
    """How many candidate strings Method A constructs for one query."""
    n = len(word)
    return n + max(n - 1, 0) + n * alphabet_size + (n + 1) * alphabet_size


def strings_built_method_b(word: str) -> int:
    """How many probe strings Method B constructs for one query."""
    return len(word) + 1  # the word itself plus one deletion per position


def _bench(label: str, fn, batch, repeats: int) -> dict:
    """Time `fn` over the whole batch, `repeats` times; keep the best run."""
    times, n_candidates = [], 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        total = 0
        for word in batch:
            total += len(fn(word))
        times.append(time.perf_counter() - t0)
        n_candidates = total
    best = min(times)
    return {
        "label": label,
        "n_words": len(batch),
        "total_seconds": best,
        "per_word_ms": best / len(batch) * 1000,
        "words_per_second": len(batch) / best,
        "all_runs": times,
        "median_seconds": statistics.median(times),
        "candidates_returned": n_candidates,
    }


def cmd_bench(args: argparse.Namespace) -> None:
    """Push one identical batch of misspellings through Method A and Method B."""
    model, test_sentences = require_models()
    print(f"Model: {model}\n")

    batch = make_misspelling_batch(test_sentences, model.vocabulary, n=args.n)
    assert len(batch) == args.n, len(batch)
    mean_len = sum(len(w) for w in batch) / len(batch)
    print(f"Batch: exactly {len(batch):,} misspelled (non-word) tokens, "
          f"mean length {mean_len:.2f}")
    print(f"  sample: {', '.join(batch[:10])}\n")

    # ---- build costs ------------------------------------------------------ #
    t0 = time.perf_counter()
    gen_a = EditDistance1Generator(model.vocabulary)
    build_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    gen_b = SymmetricDeleteIndex(model.vocabulary, verify=True)
    build_b = time.perf_counter() - t0
    gen_b_raw = SymmetricDeleteIndex(model.vocabulary, verify=False)

    print("Preprocessing (paid once, at start-up)")
    print(f"  Method A : {build_a * 1000:8.1f} ms   "
          f"(just holds a reference to the vocabulary)")
    print(f"  Method B : {build_b * 1000:8.1f} ms   "
          f"({gen_b.n_entries:,} keys, {gen_b.n_postings:,} postings)\n")

    runs = [
        _bench("Method A  (edit-distance-1 generation)", gen_a.candidates, batch, args.repeats),
        _bench("Method B  (symmetric delete + verify)", gen_b.candidates, batch, args.repeats),
        _bench("Method B  (symmetric delete, raw)", gen_b_raw.candidates, batch, args.repeats),
    ]

    width = max(len(r["label"]) for r in runs)
    print(f"{'method':<{width}}{'total':>12}{'per word':>13}"
          f"{'words/sec':>13}{'candidates':>13}")
    print("-" * (width + 51))
    for r in runs:
        print(f"{r['label']:<{width}}{r['total_seconds']:>11.4f}s"
              f"{r['per_word_ms']:>11.4f}ms{r['words_per_second']:>13,.0f}"
              f"{r['candidates_returned']:>13,}")

    a, b = runs[0], runs[1]
    speedup = a["total_seconds"] / b["total_seconds"]
    speedup_raw = a["total_seconds"] / runs[2]["total_seconds"]

    mismatches = sum(1 for w in batch if gen_a.candidates(w) != gen_b.candidates(w))
    print(f"\nOutput equivalence: Method A and Method B (verified) disagree on "
          f"{mismatches} of {len(batch):,} words.")

    built_a = sum(strings_built_method_a(w, len(ALPHABET)) for w in batch)
    built_b = sum(strings_built_method_b(w) for w in batch)
    print("\nWhy the difference - strings constructed for the whole batch")
    print(f"  Method A : {built_a:>10,}  = n + (n-1) + n|A| + (n+1)|A|, "
          f"with |A| = {len(ALPHABET)}")
    print(f"  Method B : {built_b:>10,}  = n + 1")
    print(f"  ratio    : {built_a / built_b:>10.1f}x more strings built by Method A")

    crossover = build_b / (a["total_seconds"] / len(batch) - b["total_seconds"] / len(batch))
    print(f"\nSpeed-up: Method B is {speedup:.1f}x faster than Method A "
          f"({speedup_raw:.1f}x without the edit-distance verification step).")
    print(f"Method B's {build_b * 1000:.0f} ms index build pays for itself after "
          f"about {crossover:,.0f} lookups.")

    print("\n" + CONCLUSION.format(
        speedup=speedup, speedup_raw=speedup_raw,
        alphabet=len(ALPHABET), ratio=built_a / built_b,
        build_ms=build_b * 1000, keys=gen_b.n_entries,
        postings=gen_b.n_postings, crossover=crossover,
        a_ms=a["per_word_ms"], b_ms=b["per_word_ms"],
    ))

    save_results("speed_demon", {
        "batch_size": len(batch),
        "mean_word_length": mean_len,
        "build_seconds": {"method_a": build_a, "method_b": build_b},
        "symspell_keys": gen_b.n_entries,
        "symspell_postings": gen_b.n_postings,
        "runs": runs,
        "speedup_verified": speedup,
        "speedup_raw": speedup_raw,
        "strings_built": {"method_a": built_a, "method_b": built_b},
        "output_mismatches": mismatches,
        "crossover_lookups": crossover,
    })


CONCLUSION = """\
CONCLUSION
----------
Method B is {speedup:.1f}x faster per word ({b_ms:.4f} ms vs {a_ms:.4f} ms), and the
reason is entirely structural rather than a constant-factor implementation win.

Method A's cost is tied to the size of the alphabet. For a word of length n it
must materialise n deletions, n-1 transpositions, n*|A| replacements and
(n+1)*|A| insertions - roughly 2n|A| strings, with |A| = {alphabet} here. Every
one of those strings is allocated, hashed, and probed against the vocabulary
set. Across this batch that is {ratio:.0f}x more string construction than Method B
does, and string building plus hashing is exactly where the time goes.

Method B moves that work off the query path and into a one-off preprocessing
step. Because "one deletion on each side" is enough to cover all four edit
operations, the index only has to store each vocabulary word plus its n
deletions - {keys:,} keys and {postings:,} postings here, built in {build_ms:.0f} ms. At
query time the alphabet has vanished from the complexity: only the n+1 probe
strings (the word plus its own deletions) are built, so lookup is O(n) hash
probes instead of O(n|A|). That is the whole trick - the alphabet factor is
paid once at build time instead of on every single lookup.

The verification step costs something: raw bucket lookup is {speedup_raw:.1f}x faster than
Method A, and adding the exact edit-distance-1 check brings that down to
{speedup:.1f}x. It is worth paying, because SymSpell buckets can collide at edit
distance 2 ("abc" and "axb" both delete to "ab"), so without verification
Method B returns a superset rather than the true edit-distance-1 set. With
verification the two methods return byte-identical candidate sets.

The trade is memory and start-up time for query latency. The index costs
{build_ms:.0f} ms to build and pays for itself after roughly {crossover:,.0f} lookups - so for a
one-shot correction Method A is fine, and for anything interactive or batched
Method B is the only sensible choice.
"""


# --------------------------------------------------------------------------- #
# Part 5 - output examples
# --------------------------------------------------------------------------- #

DEMO_SENTENCES = [
    # -- the four sentences from the brief --------------------------------- #
    "I hav a good feeling about this.",
    "This is a test sentnce.",
    "I would like to sea the world.",
    "Please meat me at the station.",
    # -- a few of our own --------------------------------------------------- #
    "She is a very god student in the class.",
    "The govenment anounced a new policey yesterday.",
    "He wants to by a peace of cake.",
    "Thier answer was completly diffrent from ours.",
    "I can not beleive how quikly the yeer went.",
    "The whether was terrible during the hole trip.",
]


def cmd_demo(args: argparse.Namespace) -> None:
    """Run the sample sentences non-interactively, so they can be captured."""
    _, corrector = load_models(real_word_threshold=args.threshold)
    for text in DEMO_SENTENCES:
        t0 = time.perf_counter()
        corrected, changes, _ = corrector.correct_text(text)
        latency = (time.perf_counter() - t0) * 1000
        print(f"  in  : {text}")
        print(f"  out : {corrected}")
        for c in changes:
            gain = f"  (+{c.score_gain:.2f} nats)" if c.kind == "real-word" else ""
            print(f"        [{c.kind}] {c.original} -> {c.corrected}{gain}")
        if not changes:
            print("        [no changes]")
        print(f"        latency {latency:.2f} ms\n")


# --------------------------------------------------------------------------- #
# Part 5 - the live interactive application
# --------------------------------------------------------------------------- #

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"

#: Non-word fixes and real-word fixes are highlighted differently: one is a word
#: that does not exist, the other is a judgement call about context, and the
#: user should be able to tell which is which at a glance.
KIND_COLOUR = {"non-word": GREEN, "real-word": YELLOW}


class Style:
    """ANSI colours when the terminal supports them, **asterisks** otherwise."""

    def __init__(self, colour: bool):
        self.colour = colour

    def wrap(self, text: str, *codes: str) -> str:
        return "".join(codes) + text + RESET if self.colour else text

    def highlight(self, text: str, kind: str) -> str:
        if self.colour:
            return self.wrap(text, BOLD, KIND_COLOUR.get(kind, GREEN))
        return f"**{text}**"

    def strike(self, text: str) -> str:
        return self.wrap(text, DIM, RED) if self.colour else f"~{text}~"


def render(pieces, corrections, style: Style) -> str:
    """Rebuild the corrected sentence with changed words highlighted.

    `pieces` is the (original_chunk, corrected_chunk) stream from
    `correct_text`; word chunks sit at odd indices, so the k-th word chunk is at
    index 2k+1 and maps onto correction index k.
    """
    kind_by_index = {c.index: c.kind for c in corrections}
    out = []
    for i, (original, corrected) in enumerate(pieces):
        if i % 2 == 1 and original != corrected:
            kind = kind_by_index.get((i - 1) // 2, "non-word")
            out.append(style.highlight(corrected, kind))
        else:
            out.append(corrected)
    return "".join(out)


BANNER = """\
{title}
Type a sentence and press Enter.  Type `exit` to quit, `:help` for commands.
Highlighting: {nonword} = non-word fix (unigram)   {realword} = real-word fix (bigram context)
"""

HELP = """\
  exit, quit         leave the application
  :help              this list
  :realword          toggle context-based real-word checking (currently {rw})
  :method A|B|both   candidate generator (currently {method})
  :threshold X       real-word decision margin in nats (currently {threshold})
  :stats             model and session statistics
"""


def cmd_app(args: argparse.Namespace) -> None:
    """The Part 5 continuous REPL."""
    style = Style(colour=not args.plain and (args.colour or sys.stdout.isatty()))

    t0 = time.perf_counter()
    model, corrector = load_models(method=args.method,
                                   real_word_threshold=args.threshold)
    load_ms = (time.perf_counter() - t0) * 1000

    print(BANNER.format(
        title=style.wrap("Spelling Corrector - interactive mode", BOLD, CYAN),
        nonword=style.highlight("green", "non-word"),
        realword=style.highlight("yellow", "real-word"),
    ))
    print(style.wrap(
        f"  vocabulary {len(model.vocabulary):,} types | "
        f"{model.total_tokens:,} training tokens | "
        f"{len(model.bigram_counts):,} bigrams | "
        f"method {corrector.method} | threshold {corrector.real_word_threshold}",
        DIM))
    print(style.wrap(f"  models loaded in {load_ms:.0f} ms", DIM))
    print()

    check_real_words = not args.no_realword
    n_sentences = n_corrections = 0
    total_latency = 0.0

    while True:
        try:
            line = input(style.wrap("you > ", BOLD, CYAN))
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit"}:
            break

        # ---- runtime commands ---------------------------------------------- #
        if stripped.startswith(":"):
            command, _, argument = stripped[1:].partition(" ")
            command = command.lower()
            if command == "help":
                print(HELP.format(rw="on" if check_real_words else "off",
                                  method=corrector.method,
                                  threshold=corrector.real_word_threshold))
            elif command == "realword":
                check_real_words = not check_real_words
                print(f"  real-word checking {'on' if check_real_words else 'off'}")
            elif command == "method" and argument.strip() in {"A", "B", "both"}:
                corrector.method = argument.strip()
                print(f"  candidate generator: Method {corrector.method}")
            elif command == "threshold":
                try:
                    corrector.real_word_threshold = float(argument)
                    print(f"  real-word threshold: {corrector.real_word_threshold}")
                except ValueError:
                    print("  usage: :threshold 5.0")
            elif command == "stats":
                print(f"  {model}")
                print(f"  sentences corrected this session: {n_sentences}")
                print(f"  corrections applied:              {n_corrections}")
                if n_sentences:
                    print(f"  mean latency:                     "
                          f"{total_latency / n_sentences * 1000:.2f} ms")
            else:
                print("  unknown command; try :help")
            print()
            continue

        # ---- correct -------------------------------------------------------- #
        t0 = time.perf_counter()
        corrected, corrections, pieces = corrector.correct_text(
            line, check_real_words=check_real_words
        )
        latency = time.perf_counter() - t0

        n_sentences += 1
        n_corrections += len(corrections)
        total_latency += latency

        print(style.wrap("fix > ", BOLD, GREEN) + render(pieces, corrections, style))
        if corrections:
            for c in corrections:
                detail = f" (+{c.score_gain:.2f} nats)" if c.kind == "real-word" else ""
                print("      " + style.wrap(
                    f"{c.kind:<10} {style.strike(c.original)} -> "
                    f"{style.highlight(c.corrected, c.kind)}{detail}", DIM))
        else:
            print("      " + style.wrap("no changes", DIM))
        print("      " + style.wrap(f"latency {latency * 1000:.2f} ms", DIM))
        print()

    mean = (total_latency / n_sentences * 1000) if n_sentences else 0
    print(style.wrap(f"\n{n_sentences} sentence(s), {n_corrections} correction(s), "
                     f"mean latency {mean:.2f} ms. Bye.", DIM))


# --------------------------------------------------------------------------- #
# all
# --------------------------------------------------------------------------- #

def cmd_all(args: argparse.Namespace) -> None:
    """Train, evaluate, benchmark and demo, in that order."""
    if not args.skip_train:
        banner("PART 1 - corpus and model preparation")
        args.seed = 42          # the train/test split seed
        cmd_train(args)

    banner("PART 4 - accuracy on the non-word and real-word test sets")
    args.seed = 7               # the corruption seed, independent of the split
    cmd_eval(args)

    banner('PART 4 - "Speed Demon" benchmark (Method A vs Method B)')
    cmd_bench(args)

    banner("PART 5 - output examples")
    cmd_demo(args)

    banner("done")
    print(f"  Results written to {RESULTS_PATH}.")
    print("  Launch the interactive application with:  python main.py app\n")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_,
                           description=fn.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.set_defaults(func=fn)
        return p

    # Options shared by the commands that build or consume a corrector.
    def add_corrector_opts(p):
        p.add_argument("--method", default="both", choices=["A", "B", "both"],
                       help="candidate generator (default: both, the union)")
        p.add_argument("--threshold", type=float, default=5.0,
                       help="real-word decision margin in nats (default: 5.0)")

    p = add("train", cmd_train, "Part 1 - train and pickle the models")
    p.add_argument("--k", type=float, default=0.01,
                   help="add-k smoothing constant for the bigram model")
    p.add_argument("--test-ratio", type=float, default=0.10,
                   help="fraction of Brown sentences held out for Part 4")
    p.add_argument("--seed", type=int, default=42)

    p = add("eval", cmd_eval, "Part 4 - accuracy on both test sets")
    p.add_argument("--max-cases", type=int, default=3000,
                   help="cap on test cases per set (use --full for all)")
    p.add_argument("--full", action="store_true",
                   help="use every held-out sentence")
    p.add_argument("--seed", type=int, default=7)
    add_corrector_opts(p)

    p = add("bench", cmd_bench, 'Part 4 - the 1,000-word "Speed Demon" benchmark')
    p.add_argument("--n", type=int, default=1000, help="batch size (spec: 1000)")
    p.add_argument("--repeats", type=int, default=3)

    p = add("demo", cmd_demo, "Part 5 - output examples, non-interactively")
    add_corrector_opts(p)

    p = add("app", cmd_app, "Part 5 - the live interactive application")
    p.add_argument("--plain", action="store_true",
                   help="use **asterisks** instead of ANSI colour")
    p.add_argument("--colour", "--color", dest="colour", action="store_true",
                   help="force ANSI colour even when stdout is not a terminal")
    p.add_argument("--no-realword", action="store_true",
                   help="start with real-word (context) checking switched off")
    add_corrector_opts(p)

    p = add("all", cmd_all, "train + eval + bench + demo")
    p.add_argument("--skip-train", action="store_true",
                   help="reuse the existing artifacts/brown_lm.pkl")
    p.add_argument("--quick", dest="quick", action="store_true",
                   help="evaluate on 1000 cases per set instead of all of them")
    p.set_defaults(k=0.01, test_ratio=0.10, seed=42, max_cases=1000, full=True,
                   n=1000, repeats=3)
    add_corrector_opts(p)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    # `all --quick` evaluates a subset; the eval command reads these two fields.
    if getattr(args, "quick", False):
        args.full = False
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C is a normal way to abandon a long train/eval run; report it the
        # way a shell expects (128 + SIGINT) instead of as a crash.
        print()
        raise SystemExit(130)
