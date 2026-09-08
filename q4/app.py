"""
Question 4 Streamlit app — live segmentation / spelling / grammar editor.

    cd q4
    ../.venv/bin/streamlit run app.py
"""

from __future__ import annotations

import re
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
from editor.passage import PassageToken, SampledPassage, sample_passage
from editor.pcfg import load_or_train_pcfg

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@st.cache_resource
def load_stack(k: float = DEFAULT_CONFIG.add_k):
    bi, tri = load_q4_language_models(k=k)
    pcfg = load_or_train_pcfg()
    checker = LiveChecker(config=DEFAULT_CONFIG, bigram_lm=bi, trigram_lm=tri)
    return bi, tri, pcfg, checker


def split_committed_and_partial(text: str) -> tuple[list[str], str]:
    """
    Only tokens closed by whitespace/punctuation are 'committed'.
    The trailing fragment the user is still typing is left alone.
    """
    if not text:
        return [], ""
    ends_with_break = text[-1].isspace() or text[-1] in ".!?,;:"
    words = _WORD_RE.findall(text)
    if not words:
        return [], ""
    if ends_with_break:
        return [w.lower() for w in words], ""
    return [w.lower() for w in words[:-1]], words[-1].lower()


def run_on_tokens(
    checker: LiveChecker,
    tokens: list[PassageToken],
    cfg: EditorConfig,
    *,
    show_progress: bool = False,
    delay: float = 0.0,
) -> LiveCheckResult:
    records, alerts = [], []
    accumulated: list[str] = []
    token_lats, trig_lats = [], []
    log_lines: list[str] = []
    checker.reset_grammar_stats()
    ph = st.empty() if show_progress else None
    bar = st.progress(0.0) if show_progress else None

    for i, tok in enumerate(tokens):
        if delay > 0:
            time.sleep(delay)
        rec, al, lat = checker.process_token(tok, i, accumulated)
        records.append(rec)
        alerts.extend(al)
        token_lats.append(lat)
        accumulated.extend(rec.final_words)
        for a in al:
            log_lines.append(f"[{a.kind}] {a.message}")

        if accumulated and len(accumulated) % cfg.grammar_trigger_n == 0:
            g_al, g_lat = checker.trigger_grammar_check(accumulated, i)
            alerts.extend(g_al)
            trig_lats.append(g_lat)
            for a in g_al:
                log_lines.append(f"[{a.kind}] {a.message}")

        if show_progress and bar is not None and ph is not None:
            bar.progress((i + 1) / max(len(tokens), 1))
            ph.code("\n".join(log_lines[-40:]) or "(no alerts yet)")

    if accumulated and len(accumulated) % cfg.grammar_trigger_n != 0:
        g_al, g_lat = checker.trigger_grammar_check(accumulated, max(len(tokens) - 1, 0))
        alerts.extend(g_al)
        trig_lats.append(g_lat)
        for a in g_al:
            log_lines.append(f"[{a.kind}] {a.message}")
        if show_progress and ph is not None:
            ph.code("\n".join(log_lines[-40:]) or "(no alerts yet)")

    result = LiveCheckResult(
        records=records,
        alerts=alerts,
        avg_token_latency_ms=sum(token_lats) / len(token_lats) if token_lats else 0.0,
        avg_trigger_latency_ms=sum(trig_lats) / len(trig_lats) if trig_lats else 0.0,
        corrected_words=[w for r in records for w in r.final_words],
        corrected_tags=[t for r in records for t in r.pos_tags],
    )
    # stash log for callers
    result.alerts = alerts  # noqa: already set
    st.session_state["_last_alert_log"] = log_lines
    return result


def main():
    st.set_page_config(page_title="Q4 Background Editor", layout="wide")
    st.title("Q4 — Live Segmentation, Spelling & Grammar Editor")
    st.caption(
        "Reuses trained Q1 joint decoder + Q3 Method-B corrector; "
        "Q4 owns n-grams + PCFG."
    )

    with st.sidebar:
        st.header("Config")
        p = st.slider("merge probability p", 0.0, 0.3, float(DEFAULT_CONFIG.merge_probability), 0.01)
        n = st.slider("grammar trigger N", 2, 12, int(DEFAULT_CONFIG.grammar_trigger_n))
        z_thresh = st.slider(
            "grammar z-score sensitivity",
            1.0,
            5.0,
            float(DEFAULT_CONFIG.grammar_z_threshold),
            0.1,
            help="Alert only when a window's perplexity is this many robust-sigma above the passage's running baseline. Higher = fewer alerts.",
        )
        delay = st.slider("simulate delay (s)", 0.0, 0.3, 0.05, 0.01)
        seed = st.number_input("seed (optional, 0=random)", min_value=0, value=0, step=1)
        mode = st.radio("Mode", ["Simulate passage", "Live typing"])

    cfg = EditorConfig(
        merge_probability=p,
        grammar_trigger_n=n,
        typing_delay_s=delay,
        grammar_z_threshold=z_thresh,
    )
    bi, tri, pcfg, checker = load_stack()
    checker.config = cfg

    if mode == "Simulate passage":
        if st.button("Run simulation", type="primary"):
            t0 = time.perf_counter()
            passage = sample_passage(config=cfg, seed=(seed or None))
            st.write(f"**Source:** `{passage.file_id}` — {len(passage.sentences)} sentences")
            st.write("**Stream (with merges):**", passage.plain_text)

            live = run_on_tokens(
                checker, passage.tokens, cfg, show_progress=True, delay=delay
            )
            log_lines = st.session_state.get("_last_alert_log", [])

            st.subheader("Alerts")
            st.code("\n".join(log_lines) or "(no alerts)")

            st.subheader("Corrected text")
            st.write(" ".join(live.corrected_words))
            st.write(
                f"Latency — seg+spell **{live.avg_token_latency_ms:.2f} ms/token**, "
                f"grammar **{live.avg_trigger_latency_ms:.2f} ms/trigger**, "
                f"total wall **{(time.perf_counter() - t0) * 1000:.0f} ms**"
            )

            try:
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
            except Exception as exc:
                st.warning(f"End analysis skipped: {exc}")

    else:
        st.subheader("Live typing")
        st.info(
            "Only **completed** words (after a space or punctuation) are checked. "
            "The word you are still typing is ignored until you finish it. "
            "Click **Analyze passage** for the PCFG / n-gram table."
        )
        text = st.text_area(
            "Type here",
            height=160,
            placeholder="thequick brown fox jumpsover the lazy dog ",
            key="live_text",
        )

        committed, partial = split_committed_and_partial(text)
        if partial:
            st.caption(f"Still typing: `{partial}` (not checked yet)")

        if committed:
            tokens = [
                PassageToken(text=w, gold_words=[w], is_merge=False, sentence_index=0)
                for w in committed
            ]
            try:
                live = run_on_tokens(checker, tokens, cfg, show_progress=False)
                log_lines = st.session_state.get("_last_alert_log", [])
                st.subheader("Live alerts")
                st.code("\n".join(log_lines[-60:]) or "(no alerts)")
                corrected = " ".join(live.corrected_words)
                if partial:
                    corrected = (corrected + " " + partial).strip()
                st.write("**Corrected (committed):**", " ".join(live.corrected_words))
                st.write(
                    f"Latency — seg+spell {live.avg_token_latency_ms:.2f} ms/token, "
                    f"grammar {live.avg_trigger_latency_ms:.2f} ms/trigger"
                )
                st.session_state["live_result"] = live
                st.session_state["live_committed"] = committed
            except Exception as exc:
                st.error(f"Live check failed: {exc}")

        if st.button("Analyze passage", type="primary"):
            live = st.session_state.get("live_result")
            committed = st.session_state.get("live_committed") or committed
            if not live or not committed:
                st.warning("Type some completed words first.")
            else:
                passage = SampledPassage(
                    file_id="live",
                    sentences=[live.corrected_words],
                    tokens=[
                        PassageToken(text=w, gold_words=[w], is_merge=False, sentence_index=0)
                        for w in committed
                    ],
                    merge_probability=0.0,
                )
                for r in live.records:
                    r.sentence_index = 0
                try:
                    rows = analyse_passage(passage, live, bi, tri, pcfg, cfg)
                    st.subheader("Sentence analysis")
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
                            }
                            for r in rows
                        ],
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()
