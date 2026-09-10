# Question 3 — Building, Benchmarking and Deploying an Efficient Spelling Corrector

Edit-distance-1 spelling corrector on the NLTK Brown corpus, with Method A / Method B
candidate generation, Speed Demon benchmark, and an interactive terminal app.

## Layout

```text
q3/
├── main.py                 CLI (train / eval / bench / demo / app)
├── __init__.py
├── spelling/               library package
│   ├── paths.py
│   ├── tokenize.py
│   ├── language_model.py   vocab + unigram + bigram
│   ├── candidates.py       Method A + Method B (SymSpell)
│   ├── corrector.py        non-word / real-word + load_models
│   └── evaluation.py       test sets, accuracy, speed demon
├── tests/test_spelling.py
├── artifacts/              brown_lm.pkl (trained models)
├── results/                results.json
└── REPORT.md
```

## Commands

```bash
cd q3
python main.py train          # ~5 s
python main.py eval --full    # accuracy
python main.py bench          # Speed Demon
python main.py demo
python main.py app
python main.py all --skip-train
python -m pytest -q
```

`nltk` is only needed for `train`. After `artifacts/brown_lm.pkl` exists, the rest runs without it.

## Reuse from Question 4

```python
from q1_paths import ensure_q3_on_path
ensure_q3_on_path()
from spelling import load_models

model, corrector = load_models()
corrector.correct_text("This is a test sentnce.")
corrector.method_b.candidates("sentnce")
```

## Headline results

| | |
|---|---|
| Vocabulary | 40,542 types |
| Non-word accuracy | **91.96 %** |
| Real-word accuracy | **61.37 %** |
| Method B vs A | **~9.2× faster** (this machine), identical candidates |
| Interactive latency | ~0.4–1.5 ms / sentence |

Full write-up: [REPORT.md](REPORT.md). Sample CLI outputs: `results/demo_outputs.json`.
