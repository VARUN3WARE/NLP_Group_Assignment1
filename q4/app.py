"""
Question 4 Streamlit app — live segmentation / spelling / grammar editor.

    cd q4
    ../.venv/bin/streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

Q4_ROOT = Path(__file__).resolve().parent
if str(Q4_ROOT) not in sys.path:
    sys.path.insert(0, str(Q4_ROOT))

import streamlit as st

from editor.analysis import analyse_passage, format_table
from editor.config import DEFAULT_CONFIG, EditorConfig
from editor.live_checker import LiveChecker, LiveCheckResult
from editor.ngram_lm import load_q4_language_models
from editor.passage import PassageToken, sample_passage
from editor.pcfg import load_or_train_pcfg


@st.cache_resource
def load_stack(k: float = DEFAULT_CONFIG.add_k):
    bi, tri = load_q4_language_models(k=k)
    pcfg = load_or_train_pcfg()
    checker = LiveChecker(config=DEFAULT_CONFIG, bigram_lm=bi, trigram_lm=tri)
    return bi, tri, pcfg, checker


def run_live_on_tokens(checker: LiveChecker, tokens: list[PassageToken], cfg: EditorConfig):
    records, alerts = [], []
    accumulated = []
    token_lats, trig_lats = [], []
    alert_placeholder = st.empty()
    progress = st.progress(0.0)
    log_lines = []

    for i, tok in enumerate(tokens):
        rec, al, lat = checker.process_token(tok, i, accumulated)
        records.append(rec)
        alerts.extend(al)
        token_lats.append(lat)
        accumulated.extend(rec.final_words)
        for a in al:
            log_lines.append(f"[{a.kind}] {a.message}")

        if len(accumulated) % cfg.grammar_trigger_n == 0:
            g_al, g_lat = checker.trigger_grammar_check(accumulated, i)
            alerts.extend(g_al)
            trig_lats.append(g_lat)
            for a in g_al:
                log_lines.append(f"[{a.kind}] {a.message}")

        progress.progress((i + 1) / max(len(tokens), 1))
        alert_placeholder.code("\n".join(log_lines[-40:]) or "(no alerts yet)")

    if accumulated and len(accumulated) % cfg.grammar_trigger_n != 0:
        g_al, g_lat = checker.trigger_grammar_check(accumulated, len(tokens) - 1)
        alerts.extend(g_al)
        trig_lats.append(g_lat)
        for a in g_al:
            log_lines.append(f"[{a.kind}] {a.message}")
        alert_placeholder.code("\n".join(log_lines[-40:]))

    return LiveCheckResult(
        records=records,
        alerts=alerts,
        avg_token_latency_ms=sum(token_lats) / len(token_lats) if token_lats else 0.0,
        avg_trigger_latency_ms=sum(trig_lats) / len(trig_lats) if trig_lats else 0.0,
        corrected_words=[w for r in records for w in r.final_words],
        corrected_tags=[t for r in records for t in r.pos_tags],
    ), log_lines


def main():
    st.set_page_config(page_title="Q4 Background Editor", layout="wide")
    st.title("Q4 — Live Segmentation, Spelling & Grammar Editor")
    st.caption("Reuses trained Q1 joint decoder + Q3 Method-B corrector; Q4 owns n-grams + PCFG.")

    with st.sidebar:
        st.header("Config")
        p = st.slider("merge probability p", 0.0, 0.3, float(DEFAULT_CONFIG.merge_probability), 0.01)
        n = st.slider("grammar trigger N", 2, 12, int(DEFAULT_CONFIG.grammar_trigger_n))
        delay = st.slider("simulate delay (s)", 0.0, 0.3, 0.05, 0.01)
        seed = st.number_input("seed (optional, 0=random)", min_value=0, value=0, step=1)
        mode = st.radio("Mode", ["Simulate passage", "Live typing"])

    cfg = EditorConfig(merge_probability=p, grammar_trigger_n=n, typing_delay_s=delay)
    bi, tri, pcfg, checker = load_stack()
    checker.config = cfg

    if mode == "Simulate passage":
        if st.button("Run simulation", type="primary"):
            t0 = time.perf_counter()
            passage = sample_passage(config=cfg, seed=(seed or None))
            st.write(f"**Source:** `{passage.file_id}` — {len(passage.sentences)} sentences")
            st.write("**Stream (with merges):**", passage.plain_text)

            # Respect delay by sleeping inside loop via monkeypatch of process — use run with sleep
            tokens = passage.tokens
            if delay > 0:
                # slow path for demo feel
                records, alerts, accumulated = [], [], []
                token_lats, trig_lats, log_lines = [], [], []
                ph = st.empty()
                bar = st.progress(0.0)
                for i, tok in enumerate(tokens):
                    time.sleep(delay)
                    rec, al, lat = checker.process_token(tok, i, accumulated)
                    records.append(rec)
                    alerts.extend(al)
                    token_lats.append(lat)
                    accumulated.extend(rec.final_words)
                    for a in al:
                        log_lines.append(f"[{a.kind}] {a.message}")
                    if len(accumulated) % cfg.grammar_trigger_n == 0:
                        g_al, g_lat = checker.trigger_grammar_check(accumulated, i)
                        alerts.extend(g_al)
                        trig_lats.append(g_lat)
                        for a in g_al:
                            log_lines.append(f"[{a.kind}] {a.message}")
                    bar.progress((i + 1) / len(tokens))
                    ph.code("\n".join(log_lines[-40:]) or "(no alerts)")
                if accumulated and len(accumulated) % cfg.grammar_trigger_n != 0:
                    g_al, g_lat = checker.trigger_grammar_check(accumulated, len(tokens) - 1)
                    alerts.extend(g_al)
                    trig_lats.append(g_lat)
                live = LiveCheckResult(
                    records=records,
                    alerts=alerts,
                    avg_token_latency_ms=sum(token_lats) / len(token_lats) if token_lats else 0.0,
                    avg_trigger_latency_ms=sum(trig_lats) / len(trig_lats) if trig_lats else 0.0,
                    corrected_words=[w for r in records for w in r.final_words],
                    corrected_tags=[t for r in records for t in r.pos_tags],
                )
            else:
                live, log_lines = run_live_on_tokens(checker, tokens, cfg)

            st.subheader("Corrected text")
            st.write(" ".join(live.corrected_words))
            st.write(
                f"Latency — seg+spell **{live.avg_token_latency_ms:.2f} ms/token**, "
                f"grammar **{live.avg_trigger_latency_ms:.2f} ms/trigger**, "
                f"total wall **{(time.perf_counter()-t0)*1000:.0f} ms**"
            )

            rows = analyse_passage(passage, live, bi, tri, pcfg, cfg)
            st.subheader("End-of-passage analysis")
            st.code(format_table(rows))
            st.dataframe(
                [
                    {
                        "sentence": r.text,
                        "pcfg": r.pcfg_result,
                        "bigram_lp": round(r.bigram_log_prob, 2),
                        "trigram_lp": round(r.trigram_log_prob, 2),
                        "method": r.chosen_method,
                        "verdict": r.verdict,
                        "seg_merges": r.segmentation_merges,
                        "spell_fixes": r.spelling_corrections,
                    }
                    for r in rows
                ],
                use_container_width=True,
            )

    else:
        st.subheader("Live typing")
        text = st.text_area("Type here (processed incrementally on change)", height=160)
        if text.strip():
            words = [w.lower() for w in text.split() if w.strip()]
            tokens = [
                PassageToken(text=w, gold_words=[w], is_merge=False, sentence_index=0)
                for w in words
            ]
            # Build a minimal faux passage for analysis
            from editor.passage import SampledPassage

            live, log_lines = run_live_on_tokens(checker, tokens, cfg)
            st.code("\n".join(log_lines[-50:]) or "(no alerts)")
            st.write("**Corrected:**", " ".join(live.corrected_words))
            st.write(
                f"Latency — seg+spell {live.avg_token_latency_ms:.2f} ms/token, "
                f"grammar {live.avg_trigger_latency_ms:.2f} ms/trigger"
            )

            # single-sentence passage wrapper
            passage = SampledPassage(
                file_id="live",
                sentences=[live.corrected_words],
                tokens=tokens,
                merge_probability=0.0,
            )
            # re-index records to sentence 0
            for r in live.records:
                r.sentence_index = 0
            rows = analyse_passage(passage, live, bi, tri, pcfg, cfg)
            st.subheader("Sentence analysis")
            st.dataframe(
                [
                    {
                        "sentence": r.text,
                        "pcfg": r.pcfg_result,
                        "bigram_lp": round(r.bigram_log_prob, 2),
                        "trigram_lp": round(r.trigram_log_prob, 2),
                        "method": r.chosen_method,
                        "verdict": r.verdict,
                    }
                    for r in rows
                ],
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
