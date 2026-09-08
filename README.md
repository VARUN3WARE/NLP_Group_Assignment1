# NLP Group Assignment 1

Integrated group repository for Questions 1–4.

| Question | Folder | Status |
|----------|--------|--------|
| Q1 — Word segmentation + POS tagging | [`q1/`](q1/) | Complete (English + Spanish) |
| Q2 — Transition-based dependency parser | `q2/` | In progress (separate) |
| Q3 — Spelling corrector | [`q3/`](q3/) | Complete |
| Q4 — Live background editor (Streamlit) | `q4/` | Next |

Assignment brief: [`docs/Group_Assignment_1.pdf`](docs/Group_Assignment_1.pdf)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('brown'); nltk.download('treebank'); nltk.download('punkt')"
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
python main.py train           # builds models/brown_lm.pkl
python main.py all --skip-train
python main.py app             # interactive CLI
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
│   ├── scripts/
│   ├── artifacts/       # pickled English pipeline
│   └── data/            # UD corpora (local)
├── q3/                  # spelling corrector + CLI
└── q4/                  # coming next — reuses q1 + q3
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
from spelling import load_models
model, corrector = load_models()        # loads models/brown_lm.pkl
corrector.correct_text("a test sentnce.")
```
