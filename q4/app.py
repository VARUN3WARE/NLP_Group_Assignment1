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

# Alert type styling: (emoji, color, bg)
_ALERT_STYLE = {
    "SEGMENT-ALERT": ("✂️", "#0066CC", "#E6F0FF"),
    "SPELL-ALERT": ("✏️", "#CC6600", "#FFF3E6"),
    "GRAMMAR-ALERT": ("⚠️", "#CC0000", "#FFE6E6"),
}


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
    result.alerts = alerts  # noqa: already set
    st.session_state["_last_alert_log"] = log_lines
    return result


def _render_alerts(alerts: list, max_display: int = 80):
    """Render alerts as colored badges instead of plain code block."""
    if not alerts:
        st.info("No alerts — text looks clean!")
        return

    counts = {}
    for a in alerts:
        counts[a.kind] = counts.get(a.kind, 0) + 1

    # Summary badges
    badge_cols = st.columns(len(_ALERT_STYLE))
    for col, (kind, (emoji, color, bg)) in zip(badge_cols, _ALERT_STYLE.items()):
        count = counts.get(kind, 0)
        with col:
            st.markdown(
                f"""<div style="background:{bg};border-left:4px solid {color};
                border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                <span style="font-size:20px;">{emoji}</span>
                <span style="font-size:22px;font-weight:bold;color:{color};">{count}</span>
                <span style="font-size:12px;color:#666;margin-left:6px;">{kind.replace('-ALERT','')}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    # Individual alert cards
    st.markdown('<div style="max-height:400px;overflow-y:auto;">', unsafe_allow_html=True)
    for a in alerts[:max_display]:
        emoji, color, bg = _ALERT_STYLE.get(a.kind, ("📌", "#666", "#F5F5F5"))
        st.markdown(
            f"""<div style="background:{bg};border-left:3px solid {color};
            border-radius:4px;padding:8px 12px;margin-bottom:6px;font-size:13px;">
            <strong style="color:{color};">{emoji} {a.kind.replace('-ALERT','')}</strong>
            &nbsp; {a.message}</div>""",
            unsafe_allow_html=True,
        )
    if len(alerts) > max_display:
        st.caption(f"... and {len(alerts) - max_display} more alerts")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_metrics(latency_token: float, latency_trigger: float, total_ms: float | None = None):
    """Render latency as metric cards in columns."""
    cols = st.columns(3 if total_ms else 2)
    with cols[0]:
        st.metric("Seg + Spell", f"{latency_token:.2f} ms/token")
    with cols[1]:
        st.metric("Grammar Trigger", f"{latency_trigger:.2f} ms/trigger")
    if total_ms is not None and len(cols) > 2:
        with cols[2]:
            st.metric("Total Wall Time", f"{total_ms:.0f} ms")


def _render_corrected_text(corrected_words: list[str]):
    """Render corrected text in a styled container."""
    text = " ".join(corrected_words)
    st.markdown(
        f"""<div style="background:#F8F9FA;border:1px solid #E0E0E0;border-radius:8px;
        padding:16px;font-size:15px;line-height:1.6;">
        {text}</div>""",
        unsafe_allow_html=True,
    )


def _render_analysis_table(rows):
    """Render the end-of-passage analysis as a styled dataframe."""
    st.dataframe(
        [
            {
                "#": i + 1,
                "method": r.chosen_method,
                "verdict": r.verdict,
                "seg": r.segmentation_merges,
                "sp": r.spelling_corrections,
                "bigram_lp": round(r.bigram_log_prob, 2),
                "trigram_lp": round(r.trigram_log_prob, 2),
                "pcfg": r.pcfg_result,
                "sentence": r.text,
            }
            for i, r in enumerate(rows)
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "method": st.column_config.TextColumn(width="small"),
            "verdict": st.column_config.TextColumn(width="small"),
            "seg": st.column_config.NumberColumn(width="tiny"),
            "sp": st.column_config.NumberColumn(width="tiny"),
            "bigram_lp": st.column_config.NumberColumn(width="small"),
            "trigram_lp": st.column_config.NumberColumn(width="small"),
            "pcfg": st.column_config.TextColumn(width="medium"),
            "sentence": st.column_config.TextColumn(width="large"),
        },
    )


def main():
    st.set_page_config(
        page_title="Q4 Live Editor",
        page_icon="✍️",
        layout="wide",
    )

    # --- Header ---
    st.markdown(
        """<div style="display:flex;align-items:center;gap:12px;margin-bottom:0;">
        <span style="font-size:32px;">✍️</span>
        <div>
        <h1 style="margin:0;font-size:26px;">Live Segmentation, Spelling & Grammar Editor</h1>
        <p style="margin:2px 0 0 0;color:#888;font-size:13px;">
        Q1 joint decoder &nbsp;→&nbsp; Q3 Method-B corrector &nbsp;→&nbsp;
        Q4 n-grams + PCFG</p>
        </div></div>""",
        unsafe_allow_html=True,
    )
    st.divider()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        p = st.slider("Merge probability `p`", 0.0, 0.3, float(DEFAULT_CONFIG.merge_probability), 0.01)
        n = st.slider("Grammar trigger `N`", 2, 12, int(DEFAULT_CONFIG.grammar_trigger_n))
        z_thresh = st.slider(
            "Grammar z-score sensitivity",
            1.0,
            5.0,
            float(DEFAULT_CONFIG.grammar_z_threshold),
            0.1,
            help="Alert only when a window's perplexity is this many robust-sigma above the passage's running baseline. Higher = fewer alerts.",
        )
        delay = st.slider("Simulate delay (s)", 0.0, 0.3, 0.05, 0.01)
        seed = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)

        st.markdown("---")
        mode = st.radio("**Mode**", ["📊 Simulate passage", "⌨️ Live typing"], label_before=True)

        st.markdown("---")
        st.caption("Built on Q1 (segmentation/POS) + Q3 (spelling) + Q4 (n-grams/PCFG)")

    cfg = EditorConfig(
        merge_probability=p,
        grammar_trigger_n=n,
        typing_delay_s=delay,
        grammar_z_threshold=z_thresh,
    )
    bi, tri, pcfg, checker = load_stack()
    checker.config = cfg

    if mode.startswith("📊"):
        # --- Simulate passage mode ---
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("▶ Run simulation", type="primary", use_container_width=True)

        if run_btn:
            t0 = time.perf_counter()
            passage = sample_passage(config=cfg, seed=(seed or None))

            # Source info banner
            st.markdown(
                f"""<div style="background:#E8F5E9;border-radius:8px;padding:10px 16px;
                margin-bottom:12px;">
                <strong>📖 Source:</strong> <code>{passage.file_id}</code>
                &nbsp;|&nbsp; {len(passage.sentences)} sentences
                &nbsp;|&nbsp; {len(passage.tokens)} tokens</div>""",
                unsafe_allow_html=True,
            )

            with st.expander("Raw stream (with simulated merges)", expanded=False):
                st.text(passage.plain_text)

            with st.spinner("Streaming tokens through the pipeline..."):
                live = run_on_tokens(
                    checker, passage.tokens, cfg, show_progress=True, delay=delay
                )
            log_lines = st.session_state.get("_last_alert_log", [])

            total_ms = (time.perf_counter() - t0) * 1000

            # Alerts section
            st.markdown("### 🔔 Alerts")
            _render_alerts(live.alerts)

            # Corrected text
            st.markdown("### ✅ Corrected text")
            _render_corrected_text(live.corrected_words)

            # Metrics
            st.markdown("### ⏱️ Performance")
            _render_metrics(live.avg_token_latency_ms, live.avg_trigger_latency_ms, total_ms)

            # End-of-passage analysis
            try:
                rows = analyse_passage(passage, live, bi, tri, pcfg, cfg)
                st.markdown("### 📋 End-of-passage analysis")
                with st.expander("Text table", expanded=False):
                    st.code(format_table(rows))
                _render_analysis_table(rows)
            except Exception as exc:
                st.warning(f"End analysis skipped: {exc}")

    else:
        # --- Live typing mode ---
        st.markdown("### ⌨️ Live typing")
        st.info(
            "Only **completed** words (after a space or punctuation) are checked. "
            "The word you are still typing is ignored until you finish it. "
            "Click **Analyze passage** for the PCFG / n-gram table.",
            icon="💡",
        )
        text = st.text_area(
            "Type here",
            height=120,
            placeholder="thequick brown fox jumpsover the lazy dog ",
            key="live_text",
            label_visibility="collapsed",
        )

        committed, partial = split_committed_and_partial(text)
        if partial:
            st.caption(f"✏️ Still typing: `{partial}` (not checked yet)")

        if committed:
            tokens = [
                PassageToken(text=w, gold_words=[w], is_merge=False, sentence_index=0)
                for w in committed
            ]
            try:
                live = run_on_tokens(checker, tokens, cfg, show_progress=False)
                log_lines = st.session_state.get("_last_alert_log", [])

                st.markdown("### 🔔 Live alerts")
                _render_alerts(live.alerts, max_display=30)

                st.markdown("### ✅ Corrected (committed)")
                _render_corrected_text(live.corrected_words)

                st.markdown("### ⏱️ Performance")
                _render_metrics(live.avg_token_latency_ms, live.avg_trigger_latency_ms)

                st.session_state["live_result"] = live
                st.session_state["live_committed"] = committed
            except Exception as exc:
                st.error(f"Live check failed: {exc}")

        st.markdown("---")
        if st.button("🔍 Analyze passage", type="primary"):
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
                    st.markdown("### 📋 Sentence analysis")
                    with st.expander("Text table", expanded=False):
                        st.code(format_table(rows))
                    _render_analysis_table(rows)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()
