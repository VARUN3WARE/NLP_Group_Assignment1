# Question 4 — Report

Integrated background editor: Q1 joint segmentation/POS, Q3 spelling, Q4
smoothed n-grams + PCFG constituency checking.

Canonical numbers below match `results/demo_outputs.json` from
`python main.py analyze --seed 42`, `analyze --seed 7`, and `bench`
(this machine). Exact latencies vary slightly by hardware.

## Configuration choices

| Parameter | Value | Justification |
|-----------|------:|---------------|
| Merge probability `p` | **0.08** | Spec example; yields enough spacebar-miss merges for segmentation without flooding every token |
| Grammar trigger `N` | **5** | Short enough to catch local anomalies quickly; keeps trigger cost low vs per-token seg/spell |
| Add-k for Q4 LMs | **0.01** | Same operating point as Q3; avoids zero probs without swamping ML counts |
| Spell candidates | **Method B** | Q3 showed B ≡ A on candidates and ~9× faster on this machine — required for live UI |
| Q3 real-word threshold | **5.0 nats** | Q3 default for the loaded corrector |
| Live real-word margin | **≥ 18.0 nats** | Editor override in `live_checker.py` — suppresses `leafy→leaf` / `leads→lead` / `pert→part` cascades on literary text |
| Grammar z-threshold | **2.5** | Adaptive relative outlier detection (robust z-score via median + MAD); alerts only when a window's perplexity is >2.5×MAD above the passage's own running baseline |
| Grammar warmup | **3 windows** | Minimum windows seen before z-score alerts fire |
| PCFG sentence cap | **25 tokens** | CKY/Viterbi on long Melville/Chesterton sentences is too slow for interactive use; longer → `unparseable` |

**Reuse boundary:** Q1 English trigram + beam decoder are **loaded only** for segmentation scoring. Q3 vocab/unigram/bigram + Method B are **loaded only** for spelling / real-word. Q4 trains its **own** Brown add-k bigram+trigram for grammar alerts and final scores, and its **own** PTB PCFG.

## Tagset reconciliation

Q1 emits **Brown** tags (`AT`, `BEDZ`, `PPO`, …). The PCFG lexicon uses **Penn Treebank** tags. We map with a fixed lookup in `editor/tagset.py` (e.g. `AT→DT`, `BEZ→VBZ`, `PPSS→PRP`). Unseen Brown tags fall back to `NN` after stripping Brown hyphen suffixes.

**Accuracy cost:** mapping is many-to-one and loses Brown-specific distinctions (e.g. `BE*` forms → coarse PTB verb tags). Parsing still runs over **words** (ViterbiParser lexicon); Brown→PTB is applied when packaging tagged pairs for the assignment’s reconciliation requirement. PTB-OOV surface words are projected to the known noun `something` so CKY can return a structural score (`logP=…*` in the table). Projection biases lexical probability but preserves a usable constituency signal.

## Decision rule (Part 4)

1. If PCFG returns a finite log-probability and it is not a high-z outlier vs other sentences in the passage → choose **pcfg**, verdict **grammatical**.
2. Else if trigram log-prob is finite → choose **trigram**, verdict **plausible** if `tri_lp > -80` else **implausible**.
3. Else → **bigram** with the same threshold rule.

## Sample run A (`--seed 42`, bible-kjv)

Source: `bible-kjv.txt`, 5 sentences, 63 tokens.

| Alert kind | Count |
|------------|------:|
| SEGMENT | 6 |
| SPELL | 0 |
| GRAMMAR | 4 |

Latency: **0.08 ms/token** seg+spell, **0.06 ms/trigger** grammar.

- True merges fixed: `himout`, `couldnot`, `andperverse`, `devilthrew`.
- False OOV chops (limitation): `besought→be+sought`, `faithless→faith+less`.
- Grammar z-score fired on 4 anomalous windows only (not every window).
- End table: 2 short/projected sentences **pcfg / grammatical**; longer KJV lines **trigram / implausible**.

Full table: `results/demo_outputs.json` → `sample_A.table`.

## Sample run B (`--seed 7`, chesterton-thursday)

Source: `chesterton-thursday.txt`, 6 sentences, 130 tokens.

| Alert kind | Count |
|------------|------:|
| SEGMENT | 15 |
| SPELL | 0 |
| GRAMMAR | 5 |

Latency: **0.12 ms/token** seg+spell, **0.05 ms/trigger** grammar.

- Many true merges split cleanly (`wasmaking`, `ofthe`, `tosilence`, `undera`, …).
- Zero SPELL-ALERT with conservative filters.
- One short sentence **pcfg / grammatical**; long literary sentences fall to trigram.

Full table: `results/demo_outputs.json` → `sample_B.table`.

## Speed Demon (1000 words)

From `results/speed_demon.json` (this machine):

| Path | Total | Per word |
|------|------:|--------:|
| Segmentation + spelling | 0.716 s | **0.716 ms** |
| Grammar-trigger only | 0.010 s | **0.010 ms** |
| Overhead (seg+spell − grammar) | — | **0.706 ms** |

**Conclusion:** The segmentation+spelling layer is still well under a few ms/word — cheap enough to keep on every token in live mode; throttling it to the grammar interval is unnecessary.

## Comparative analysis

1. **Live alerts vs final verdict:** SEGMENT/SPELL alerts are local and usually agree with the corrected sentence used in Part 4. GRAMMAR-ALERT windows fire adaptively via robust z-score against the passage’s own baseline, so Brown-vs-Gutenberg domain shift no longer floods every window.

2. **PCFG vs n-grams:** PCFG (when it parses) prefers short, PTB-like structure. N-grams catch local lexical implausibility and domain shift. Long literary sentences almost never get a PCFG verdict under the length cap — n-grams carry those cases.

3. **`p` and `N`:** Higher `p` increases SEGMENT-ALERT rate and gives Q1 real work; higher `N` delays GRAMMAR-ALERT but reduces trigger overhead. `p=0.08`, `N=5` is a practical balance.

4. **Sub-system interactions:** A successful SEGMENT split can turn an OOV merge into in-vocab words and suppress SPELL. Conversely, a SPELL rewrite can flip window perplexity and the Part-4 method. PCFG projection (`*`) alters lexical identity for parseability — another interaction when comparing methods. Live real-word margin (≥18) is stricter than Q3’s default so the editor does not cascade literary rewrites into grammar noise.

## Known limitations

1. **Literary / archaic OOV words:** Absent from Brown (e.g. `hatter`, `dormouse`, `besought`). Conservative spell filters reject first-letter changes and weak shared stems; some genuine OOVs stay uncorrected rather than being wrongly rewritten.

2. **Over-segmentation of morphologically complex words:** OOV words whose substrings are valid (`besought`, `faithless`) may be incorrectly split — the spacebar-miss model cannot always tell one morphological word from two merged words.

3. **Binary split fallback:** When Q1 over-segments (`saidthank→said+than+k`), Q4 tries a binary in-vocab cut (`said+thank`, `march+hare`, `asit's`). This fails if either piece is missing from Brown.

## Streamlit (Part 5)

```bash
cd q4 && ../.venv/bin/streamlit run app.py
```

Modes: **Simulate passage** (auto stream with merges) and **Live typing** (only completed words after space/punctuation). Alerts use color-coded cards; end analysis shows the Part-4 table and latency metrics. CLI transcripts for fixed seeds are in `results/demo_outputs.json` (Streamlit mirrors the same pipeline).

## Submission checklist (PDF Parts 1–5)

| Part | Evidence |
|------|----------|
| 1 Live stream + merges + SEGMENT/SPELL/GRAMMAR | `editor/passage.py`, `live_checker.py`; `analyze --seed 42/7` |
| 2 PCFG + Viterbi + unparseable + Brown→PTB | `editor/pcfg.py`, `tagset.py`; table `pcfg` column |
| 3 Q4 bigram/trigram (reuse Q1/Q3, no retrain) | `editor/ngram_lm.py`, `adapters.py` |
| 4 End table + decision rule | `editor/analysis.py`; `demo_outputs.json` tables |
| 5 Streamlit + Speed Demon | `app.py`; `python main.py bench` → `speed_demon.json` |

## How to reproduce

```bash
cd q4
python main.py train-lms                 # once
python main.py analyze --seed 42
python main.py analyze --seed 7
python main.py bench
streamlit run app.py
```
