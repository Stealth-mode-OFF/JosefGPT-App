from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import streamlit as st
from app.llm import get_chat_llm
from app.query_engine import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    answer_with_context,
)

st.set_page_config(page_title="JosefGPT Local", layout="wide")
st.title("🧠 JosefGPT — Hybrid Reasoning Chat")

llm = get_chat_llm()

if "history" not in st.session_state:
    st.session_state.history = []
elif st.session_state.history and isinstance(st.session_state.history[0], tuple):
    st.session_state.history = [
        {"question": q, "answer": a, "sources": [], "contexts": [], "config": None, "llm": None}
        for q, a in st.session_state.history
    ]
settings_defaults = {
    "top_k": DEFAULT_TOP_K,
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
}
if "settings" not in st.session_state:
    st.session_state.settings = settings_defaults.copy()

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**LLM mode:** `{llm.mode}` · `{llm.model_name}`")
    if llm.mode == "offline":
        st.info("Offline mode is active. Answers use local heuristics.")
    top_k = st.slider(
        "Context chunks (k)",
        min_value=1,
        max_value=12,
        value=int(st.session_state.settings.get("top_k", DEFAULT_TOP_K)),
        help="How many retrieved chunks to send to the model.",
    )
    temperature = st.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.settings.get("temperature", DEFAULT_TEMPERATURE)),
        step=0.05,
        help="Higher values = more exploratory answers.",
    )
    max_tokens = st.slider(
        "Answer length (max tokens)",
        min_value=200,
        max_value=2000,
        value=int(st.session_state.settings.get("max_tokens", DEFAULT_MAX_TOKENS)),
        step=50,
        help="Upper bound on generated token count.",
    )
    if st.button("Reset to defaults"):
        st.session_state.settings = settings_defaults.copy()
        top_k = settings_defaults["top_k"]
        temperature = settings_defaults["temperature"]
        max_tokens = settings_defaults["max_tokens"]

st.session_state.settings.update(
    {"top_k": top_k, "temperature": temperature, "max_tokens": max_tokens}
)

if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if st.session_state.pop("_reset_question_input", False):
    st.session_state.question_input = ""

user_input = st.text_area("💬 Your question", key="question_input", height=100)

if st.button("Ask") and user_input.strip():
    with st.spinner("Thinking..."):
        result = answer_with_context(
            user_input,
            top_k=st.session_state.settings["top_k"],
            temperature=st.session_state.settings["temperature"],
            max_tokens=st.session_state.settings["max_tokens"],
        )
    entry = {
        "question": user_input,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "contexts": result.get("contexts", []),
        "config": result.get("config"),
        "llm": result.get("llm"),
    }
    st.session_state.history.append(entry)
    st.session_state["_reset_question_input"] = True
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - fallback for older Streamlit versions
        st.experimental_rerun()

for item in reversed(st.session_state.history):
    st.markdown(f"**🧍‍♂️ You:** {item.get('question', '')}")
    st.markdown(f"**🤖 JosefGPT:** {item.get('answer', '')}")
    config = item.get("config") or {}
    llm_info = item.get("llm") or {}
    meta_bits = []
    if "top_k" in config:
        meta_bits.append(f"k={int(config['top_k'])}")
    if "temperature" in config:
        meta_bits.append(f"temp={config['temperature']:.2f}")
    if "max_tokens" in config:
        meta_bits.append(f"max_tokens={int(config['max_tokens'])}")
    if llm_info:
        meta_bits.append(f"llm={llm_info.get('mode', '?')} ({llm_info.get('model', '?')})")
    if meta_bits:
        st.caption(" • ".join(meta_bits))
    sources = item.get("sources") or []
    if sources:
        st.markdown("**📚 Supporting context:**")
        for source in sources:
            label = source.get("source", "Unknown source")
            chunk = source.get("chunk")
            score = source.get("score")
            details = []
            if chunk is not None:
                details.append(f"chunk {chunk}")
            if isinstance(score, (int, float)):
                details.append(f"score {score:.2f}")
            expander_title = label if not details else f"{label} ({', '.join(details)})"
            with st.expander(expander_title):
                st.write(source.get("preview", ""))
    st.markdown("---")
