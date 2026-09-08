# NLP Group Assignment 1

Integrated group repository for Questions 1–4.

| Question | Folder | Status |
|----------|--------|--------|
| Q1 — Word segmentation + POS tagging | [`q1/`](q1/) | Complete (English + Spanish) |
| Q2 — Transition-based dependency parser | `q2/` | In progress (separate) |
| Q3 — Spelling corrector | [`q3/`](q3/) | Complete |
| Q4 — Live background editor (Streamlit) | [`q4/`](q4/) | Complete |

Assignment brief: [`docs/Group_Assignment_1.pdf`](docs/Group_Assignment_1.pdf)

## Quick start

```bash
cd Group_Assignment_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# NLTK data lives in ~/nltk_data (private path)
python -c "import nltk; [nltk.download(p) for p in ('brown','treebank','punkt','gutenberg','reuters')]"
```

Use this project's `.venv` for all commands from here on:

```bash
source .venv/bin/activate
# or call directly: .venv/bin/python ...
```

### Question 1

```bash
cd q1
# Optional: clone UD Spanish-GSD into data/spanish/UD_Spanish-GSD/
python main.py train-english   # -> artifacts/english_pipeline.pkl
python main.py sample
python main.py evaluate        # full metrics (slow)
```

### Question 3

```bash
cd q3
python main.py train           # -> artifacts/brown_lm.pkl
python main.py all --skip-train
python main.py app             # interactive CLI
```

### Question 4

```bash
cd q4
python main.py train-lms              # Q4 bigram/trigram + PCFG cache
python main.py analyze --seed 42      # sample run + results JSON
python main.py bench                  # 1000-word Speed Demon
streamlit run app.py                  # live UI
```

## Layout

```text
.
├── README.md
├── requirements.txt
├── docs/
│   └── Group_Assignment_1.pdf
├── q1/
│   ├── main.py
│   ├── segpos/          # package (data, lm, segmentation, tagging, eval)
│   ├── artifacts/       # pickled English pipeline
│   └── data/            # UD corpora (local)
├── q3/
│   ├── main.py
│   ├── spelling/        # package (lm, candidates, corrector, evaluation)
│   ├── tests/
│   ├── artifacts/       # brown_lm.pkl
│   └── results/
├── q4/
│   ├── main.py / app.py
│   ├── editor/          # live checker, ngrams, pcfg, analysis
│   ├── artifacts/
│   └── results/
```

## Cross-question reuse (for Q4)

**From Q1 (English only):**

```python
from q1_paths import ensure_q1_on_path
ensure_q1_on_path()
from segpos import load_english_pipeline
pipe = load_english_pipeline()          # no retrain
pipe.decode("thequickbrownfox")         # joint beam → [(word, tag), ...]
```

**From Q3:**

```python
from q1_paths import ensure_q3_on_path
ensure_q3_on_path()
from spelling import load_models
model, corrector = load_models()        # loads artifacts/brown_lm.pkl
corrector.correct_text("a test sentnce.")
```
