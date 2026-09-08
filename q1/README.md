# Question 1 — Word Segmentation and POS Tagging

English + Spanish segmentation and POS tagging with trigram models, Viterbi
decoding, baselines, and a joint beam decoder for Q4 reuse.

## Layout

```text
q1/
├── main.py                      CLI entrypoint
├── README.md
├── segpos/                      importable package
│   ├── paths.py                 q1 root / artifacts / data paths
│   ├── pipeline.py              train / save / load English bundle
│   ├── data/
│   │   └── corpus.py            Brown + Spanish CoNLL-U
│   ├── lm/
│   │   └── trigram.py           trigram word LM
│   ├── segmentation/
│   │   ├── viterbi.py           Viterbi segmenter
│   │   └── joint_beam.py        joint beam decode (α, β, beam)
│   ├── tagging/
│   │   └── pos_tagger.py        HMM POS + morphology-aware Spanish
│   ├── baselines/
│   │   └── simple.py            greedy segment + most-frequent tag
│   └── eval/
│       ├── metrics.py           accuracy, confusion, error sources
│       └── experiments.py       full EN/ES evaluation runner
├── scripts/
│   ├── evaluate.py              python scripts/evaluate.py
│   └── sample_outputs.py
├── artifacts/                   english_pipeline.pkl (gitignored)
└── data/                        UD corpora (gitignored)
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
python main.py evaluate        # full metrics (slow)
python scripts/evaluate.py     # same evaluation
```

## Reuse from Question 4

Run with `q1/` on `PYTHONPATH`, or from inside `q1/`:

```python
from segpos import load_english_pipeline

pipe = load_english_pipeline()
pipe.decode("thequickbrownfox")
split, pairs = pipe.should_split("thequick")
```

Defaults stored in the pickle: `max_word_length=20`, `α=1.0`, `β=1.0`, `beam_width=8`.

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
