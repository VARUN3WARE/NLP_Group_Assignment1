# Question 3 — Building, Benchmarking and Deploying an Efficient Spelling Corrector

An edit-distance-1 spelling corrector trained on the NLTK Brown corpus, with two
independent candidate-generation algorithms, a benchmark that pits them against
each other, and an interactive terminal application.

## Layout

Three Python files.

```
q3/
├── spelling.py         the library — everything the corrector needs
│                         Part 1  LanguageModel            vocabulary, unigram, bigram
│                         Part 2  EditDistance1Generator   Method A
│                                 SymmetricDeleteIndex     Method B (SymSpell)
│                         Part 3  SpellingCorrector        non-word + real-word
│                         Part 4  build_test_sets, evaluate, speed_demon
├── main.py             the drivers, one subcommand per part
├── test_spelling.py    unit tests
├── models/             brown_lm.pkl, test_split.pkl   (generated)
├── results/            results.json, full_run.txt      (generated)
└── REPORT.md           the write-up, with all the numbers
```

## Running it

`nltk` is needed **only** to train. Once `models/brown_lm.pkl` exists, everything
else runs on the standard library alone.

```bash
python main.py train          # ~5 s   Part 1  train and pickle the models
python main.py eval --full    # ~45 s  Part 4  accuracy on both test sets
python main.py bench          # ~2 s   Part 4  the 1,000-word Speed Demon benchmark
python main.py demo           #        Part 5  output examples, non-interactively
python main.py app            #        Part 5  the live interactive application
python main.py all            # ~50 s  train + eval + bench + demo
python -m pytest -q           # ~2 s   the test suite
```

`python main.py <command> --help` lists that command's options. The Brown corpus
is fetched once with `nltk.download("brown")`.

Run commands from inside `q3/` so local imports resolve cleanly.

### Interactive application

```
$ python main.py app
you > This is a test sentnce.
fix > This is a test sentence.
      non-word   sentnce -> sentence
      latency 0.24 ms
you > exit
```

Changed words are highlighted in colour — green for a non-word fix, yellow for a
context-driven real-word fix — falling back to `**asterisks**` with `--plain` or
when stdout is not a terminal. Runtime commands: `:help`, `:realword`,
`:method A|B|both`, `:threshold X`, `:stats`.

## Headline results

| | |
|---|---|
| Vocabulary | 40,542 types over 901,816 training tokens |
| Non-word correction accuracy | **91.96 %** (5,620 held-out cases) |
| Real-word correction accuracy | **61.37 %** (same 5,620 sentences) |
| Method B vs Method A | **13.6× faster** per word, byte-identical output |
| Interactive latency | ~0.4 ms per sentence |

Full analysis, including the accuracy/false-alarm trade-off and the failure
modes, is in [REPORT.md](REPORT.md).

## Reuse from Question 4

```python
from spelling import load_models

model, corrector = load_models()          # loads models/brown_lm.pkl, no retraining
corrector.correct_text("This is a test sentnce.")
model.log_bigram_prob("to", "see")        # add-k smoothed
corrector.method_b.candidates("sentnce")  # Method B directly
```
