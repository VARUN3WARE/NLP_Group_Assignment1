# Question 1 — Word Segmentation and POS Tagging

English + Spanish segmentation and POS tagging with trigram models, Viterbi
decoding, baselines, and a joint beam decoder for Q4 reuse.

## Layout

```text
q1/
├── main.py                 CLI (train-english / sample / evaluate)
├── __init__.py             allows `import q1.segpos` from repo root
├── segpos/                 library package
│   ├── paths.py
│   ├── pipeline.py         train / save / load English bundle
│   ├── data/corpus.py
│   ├── lm/trigram.py
│   ├── segmentation/       viterbi.py, joint_beam.py
│   ├── tagging/pos_tagger.py
│   ├── baselines/simple.py
│   └── eval/               metrics.py, experiments.py
├── artifacts/              english_pipeline.pkl (gitignored)
└── data/                   UD corpora (gitignored)
```

## Setup

```bash
cd q1
pip install nltk
python -c "import nltk; nltk.download('brown')"

# Spanish (optional for English-only / Q4)
mkdir -p data/spanish
git clone https://github.com/UniversalDependencies/UD_Spanish-GSD.git \
  data/spanish/UD_Spanish-GSD
```

## Commands

```bash
python main.py train-english   # -> artifacts/english_pipeline.pkl
python main.py sample
python main.py evaluate        # full metrics (slow; needs Spanish data)
```

## Reuse from Question 4

```python
from q1_paths import ensure_q1_on_path
ensure_q1_on_path()
from segpos import load_english_pipeline

pipe = load_english_pipeline()
pipe.decode("thequickbrownfox")
split, pairs = pipe.should_split("thequick")
```

Defaults in the pickle: `max_word_length=20`, `α=1.0`, `β=1.0`, `beam_width=8`.

## Results (summary)

| Task | Model | Baseline | Δ |
|------|------:|--------:|--:|
| EN segmentation | 48.85% | 25.50% | +23.35 pp |
| ES segmentation | 14.99% | 6.09% | +8.90 pp |
| EN POS | 92.94% | 87.29% | +5.65 pp |
| ES POS | 92.94% | 88.55% | +4.39 pp |
| ES morph POS | 90.57% | — | −2.38 pp vs plain |

English end-to-end errors: ~66% from segmentation, ~34% genuine tagging.

English uses the **Brown** tagset (`AT`, `NN`, `JJ`, …).
