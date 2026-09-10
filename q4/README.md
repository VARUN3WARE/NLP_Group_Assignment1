# Question 4 — Integrated Background Editor

Live segmentation (Q1) + spelling (Q3) + grammar/PCFG analysis in one editor.

See **[REPORT.md](REPORT.md)** for configuration justifications, sample runs,
Speed Demon, and the comparative analysis. Canonical numbers:
`results/demo_outputs.json`.

## Layout

```text
q4/
├── main.py              CLI: train-lms / simulate / analyze / bench
├── app.py               Streamlit live + simulate UI
├── REPORT.md
├── editor/
│   ├── adapters.py      load Q1/Q3 once (no retrain)
│   ├── passage.py       random Gutenberg paragraph + merge noise
│   ├── live_checker.py  SEGMENT / SPELL / GRAMMAR alerts
│   ├── ngram_lm.py      Q4-owned add-k bigram + trigram
│   ├── tagset.py        Brown → PTB map
│   ├── pcfg.py          induce PCFG + Viterbi/CKY parse
│   ├── analysis.py      end-of-passage decision table
│   └── benchmark.py     1000-word Speed Demon
├── artifacts/           q4_*.pkl (gitignored)
└── results/             demo_outputs.json, run_*.json, speed_demon.json
```

## Setup

```bash
cd Group_Assignment_1
source .venv/bin/activate

# Ensure Q1 English pickle exists
cd q1 && python main.py train-english && cd ..

# Train Q4 LMs + PCFG cache
cd q4
python main.py train-lms
```

## Commands

```bash
cd q4
python main.py simulate --seed 42          # stream alerts (no save)
python main.py analyze --seed 42           # + save results JSON
python main.py analyze --seed 7
python main.py bench                       # Speed Demon (1000 words)
streamlit run app.py                       # live UI (simulate + typing)
```

## Headline results (this machine)

| | |
|---|---|
| Seed 42 alerts | 6 SEGMENT / 0 SPELL / 4 GRAMMAR |
| Seed 7 alerts | 15 SEGMENT / 0 SPELL / 5 GRAMMAR |
| Seg+spell latency | ~0.08–0.12 ms/token (live); **0.72 ms/word** (bench) |
| Grammar trigger | ~0.05–0.06 ms/trigger; **0.01 ms/word** (bench) |

## Defaults (see REPORT.md)

| Knob | Value |
|------|------:|
| merge `p` | 0.08 |
| grammar `N` | 5 |
| grammar z | 2.5 |
| add-k | 0.01 |
| spell method | B |
| live real-word margin | ≥ 18 nats |
| PCFG max length | 25 tokens |

## Reuse

```python
# inside q4 — adapters already do this
from editor.adapters import get_q1_pipeline, get_q3_models
pipe = get_q1_pipeline()          # Q1 joint beam
model, corrector = get_q3_models()  # Q3 Method B
```
