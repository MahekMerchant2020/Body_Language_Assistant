from __future__ import annotations

import importlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# =========================
# Safe root detection
# =========================

APP_DIR = Path(__file__).resolve().parent

if (APP_DIR / "retrieval").exists():
    PROJECT_ROOT = APP_DIR
elif (APP_DIR.parent / "retrieval").exists():
    PROJECT_ROOT = APP_DIR.parent
else:
    raise RuntimeError(
        f"Could not find a 'retrieval' folder relative to app.py. "
        f"Checked: {APP_DIR} and {APP_DIR.parent}"
    )

TMP_UPLOAD_DIR = APP_DIR / ".tmp_uploads"
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================
# Page config
# =========================

st.set_page_config(
    page_title="Body Language Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# Session state
# =========================

defaults = {
    "chat_history": [],
    "graph": None,
    "backend_key_sig": None,
    "error_message": "",
    "message_box": "",
    "panel_open": True,
    "is_processing": False,
    "pending_query": None,
    "pending_image_path": None,
    "openrouter_key": os.environ.get("OPENROUTER_API_KEY", ""),
    "tavily_key": os.environ.get("TAVILY_API_KEY", ""),
    "uploader_version": 0,
    "uploader_key": "main_uploader_0",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# Styling
# =========================

st.markdown(
    """
    <style>
    :root {
        --bg: #F5F3EE;
        --surface: #FBFAF7;
        --surface-2: #F0ECE4;
        --text: #161616;
        --muted: #5F625C;
        --border: #D8D2C7;
        --accent: #2F4A3F;
        --accent-hover: #243A31;
        --focus: #6C8B7A;
        --error-text: #8B2E34;
        --error-bg: #F8EDEE;
        --error-border: #E3C5C8;
    }

    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: Inter, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
    }

    .block-container {
        max-width: 1460px;
        padding-top: 22px;
        padding-left: 24px;
        padding-right: 24px;
        padding-bottom: 28px;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: var(--text);
    }

    .app-title {
        font-family: Inter, "SF Pro Display", "Helvetica Neue", Arial, sans-serif !important;
        font-size: 40px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text) !important;
        margin: 0 0 6px 0;
        line-height: 1.08;
    }

    .app-subtitle {
        font-size: 15px;
        font-weight: 400;
        color: var(--muted) !important;
        margin: 0 0 20px 0;
    }

    .soft-divider {
        height: 1px;
        background: var(--border);
        margin: 0 0 22px 0;
    }

    .section-heading {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--accent) !important;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    .helper-text {
        font-size: 12px;
        line-height: 1.6;
        color: var(--text) !important;
    }

    .source-text {
        font-size: 13px;
        line-height: 1.6;
        color: var(--text) !important;
        margin-bottom: 8px;
    }

    .chat-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted) !important;
        margin: 0 0 6px 2px;
    }

    .msg-user, .msg-assistant {
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
        color: var(--text) !important;
        line-height: 1.65;
        margin-bottom: 16px;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .msg-user {
        background: var(--surface-2);
    }

    .msg-assistant {
        background: var(--surface);
        padding: 16px 18px;
    }

    .meta-row {
        color: var(--muted) !important;
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.5;
    }

    .status-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        max-width: 720px;
        margin-top: 10px;
    }

    .status-text {
        font-size: 15px;
        color: var(--muted) !important;
        line-height: 1.6;
    }

    .upload-helper-inline {
        font-size: 12px;
        color: var(--muted) !important;
        margin-top: 4px;
        margin-bottom: 10px;
    }

    .error-panel {
        background: var(--error-bg);
        border: 1px solid var(--error-border);
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 14px;
        color: var(--error-text) !important;
        font-size: 13px;
        font-weight: 500;
        line-height: 1.55;
    }

    .stTextInput input,
    .stTextArea textarea,
    textarea,
    input {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        caret-color: #161616 !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus,
    textarea:focus,
    input:focus {
        border-color: var(--focus) !important;
        box-shadow: 0 0 0 2px rgba(108, 139, 122, 0.16) !important;
        color: var(--text) !important;
        caret-color: #161616 !important;
    }

    .stTextArea textarea {
        min-height: 96px !important;
    }

    .stButton > button {
        background: #EAE6DE !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        min-height: 42px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #DDD7CD !important;
        color: var(--text) !important;
        border-color: var(--accent) !important;
    }

    .stFileUploader {
        margin-top: 4px !important;
        margin-bottom: 8px !important;
    }

    .stFileUploader > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 14px !important;
        padding: 0.7rem 0.95rem !important;
        min-height: auto !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--muted) !important;
        font-size: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: transparent !important;
        color: var(--accent) !important;
        border: none !important;
        box-shadow: none !important;
    }

    .stAlert {
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Backend loading
# =========================

def _reload_backend_modules() -> Any:
    module_names = [
        "retrieval.classifier",
        "retrieval.query_enhancer",
        "retrieval.reranker",
        "retrieval.vector_search",
        "retrieval.hybrid_search",
        "retrieval.answer_generator",
        "retrieval.web_agent",
        "retrieval.image_interpreter",
        "retrieval.orchestrator",
    ]

    for name in module_names:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)

    orchestrator = sys.modules["retrieval.orchestrator"]
    return orchestrator.build_graph()


def _get_graph(openrouter_key: str, tavily_key: str):
    key_sig = f"{openrouter_key.strip()}|{tavily_key.strip()}"

    if st.session_state.graph is not None and st.session_state.backend_key_sig == key_sig:
        return st.session_state.graph

    os.environ["OPENROUTER_API_KEY"] = openrouter_key.strip()
    os.environ["TAVILY_API_KEY"] = tavily_key.strip()

    graph = _reload_backend_modules()
    st.session_state.graph = graph
    st.session_state.backend_key_sig = key_sig
    return graph


# =========================
# Helpers
# =========================

def _save_uploaded_file(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        st.session_state.error_message = "Only .jpg, .jpeg, and .png files are accepted."
        return None

    target = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def _set_example(text: str) -> None:
    st.session_state.message_box = text


def _render_title_block() -> None:
    st.markdown('<div class="app-title">Body Language Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">A research-driven interface for interpreting nonverbal behavior</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)


def _render_prekey_state() -> None:
    st.markdown(
        """
        <div class="status-panel">
            <div class="status-text">
                Please enter API keys in the left panel.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chat_history() -> None:
    for msg in st.session_state.chat_history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        extra = msg.get("meta")

        label = "You" if role == "user" else "Assistant"
        klass = "msg-user" if role == "user" else "msg-assistant"

        st.markdown(f'<div class="chat-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{klass}">{content}</div>', unsafe_allow_html=True)

        if role == "assistant" and extra:
            parts = []
            if extra.get("source"):
                parts.append(f"Source: {extra['source']}")
            if extra.get("retrieval_confidence") is not None:
                parts.append(f"Retrieval confidence: {extra['retrieval_confidence']}")
            if extra.get("answerability_confidence") is not None:
                parts.append(f"Answerability confidence: {extra['answerability_confidence']}")
            if parts:
                st.markdown(f'<div class="meta-row">{" · ".join(parts)}</div>', unsafe_allow_html=True)


def _build_user_payload(query: str, image_path: Optional[str]) -> Dict[str, Any]:
    return {
        "user_query": query,
        "image_path": image_path,
        "chat_history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ],
    }


# =========================
# Left panel
# =========================

def _render_left_panel() -> None:
    with st.container(border=True):
        head_left, head_right = st.columns([5, 1], gap="small")
        with head_right:
            if st.button("<", key="close_panel_btn", use_container_width=True):
                st.session_state.panel_open = False
                st.rerun()

        st.markdown('<div class="section-heading">API Configuration</div>', unsafe_allow_html=True)
        st.text_input(
            "OpenRouter API Key",
            type="password",
            key="openrouter_key",
            disabled=st.session_state.is_processing,
        )
        st.text_input(
            "Tavily API Key",
            type="password",
            key="tavily_key",
            disabled=st.session_state.is_processing,
        )
        st.caption("Keys are used only for this active session.")

        st.divider()

        st.markdown('<div class="section-heading">Query Examples</div>', unsafe_allow_html=True)
        if st.button("Interpret the body language in this image.", key="example_0", use_container_width=True, disabled=st.session_state.is_processing):
            _set_example("Interpret the body language in this image.")
            st.rerun()

        if st.button("What does crossed arms and averted gaze usually suggest?", key="example_1", use_container_width=True, disabled=st.session_state.is_processing):
            _set_example("What does crossed arms and averted gaze usually suggest?")
            st.rerun()

        if st.button("What is the latest research on microexpressions?", key="example_2", use_container_width=True, disabled=st.session_state.is_processing):
            _set_example("What is the latest research on microexpressions?")
            st.rerun()

        st.divider()

        st.markdown('<div class="section-heading">Image Upload Guidance</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="helper-text">Supported formats: .jpg, .jpeg, .png<br>'
            'Use a clear image with visible face, hands, and posture where possible.<br>'
            'Avoid heavily cropped, blurred, or low-light images.</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown('<div class="section-heading">Sources Available</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-text">The Definitive Book of Body Language — Allan &amp; Barbara Pease</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-text">Web Search</div>', unsafe_allow_html=True)


# =========================
# Request processing
# =========================

def _run_pending_request(openrouter_key: str, tavily_key: str) -> None:
    if not st.session_state.is_processing or not st.session_state.pending_query:
        return

    query = st.session_state.pending_query
    image_path = st.session_state.pending_image_path

    with st.spinner("Analyzing body language..."):
        try:
            graph = _get_graph(openrouter_key, tavily_key)
            payload = _build_user_payload(query, image_path)
            result = graph.invoke(payload)

            final_answer = result.get("final_answer", {}) or {}
            answer_text = final_answer.get("answer", "I could not generate a response.")

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "meta": {
                        "source": final_answer.get("source"),
                        "retrieval_confidence": final_answer.get("retrieval_confidence"),
                        "answerability_confidence": final_answer.get("answerability_confidence"),
                    },
                }
            )

        except Exception as e:
            err = str(e)

            if "OPENROUTER" in err.upper() or "401" in err or "403" in err:
                st.session_state.error_message = "OpenRouter API key appears invalid. Please review and try again."
            elif "TAVILY" in err.upper():
                st.session_state.error_message = "Tavily request failed. Check your API key or network access."
            else:
                st.session_state.error_message = f"An unexpected error occurred while generating a response: {err}"

        finally:
            st.session_state.is_processing = False
            st.session_state.pending_query = None
            st.session_state.pending_image_path = None
            st.session_state.message_box = ""
            st.session_state.uploader_version += 1
            st.session_state.uploader_key = f"main_uploader_{st.session_state.uploader_version}"
            st.rerun()


# =========================
# Main area
# =========================

def _render_main() -> None:
    if not st.session_state.panel_open:
        if st.button("Open panel", key="open_panel_btn", disabled=st.session_state.is_processing):
            st.session_state.panel_open = True
            st.rerun()

    _render_title_block()

    openrouter_key = st.session_state.openrouter_key
    tavily_key = st.session_state.tavily_key
    keys_ready = bool(openrouter_key.strip()) and bool(tavily_key.strip())

    if not keys_ready:
        _render_prekey_state()
        return

    _run_pending_request(openrouter_key, tavily_key)

    _render_chat_history()

    uploaded_file = st.file_uploader(
        "Attach an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=st.session_state.uploader_key,
        help="Only .jpg, .jpeg, and .png files are accepted.",
        disabled=st.session_state.is_processing,
    )

    st.markdown(
        '<div class="upload-helper-inline">Accepted formats: .jpg, .jpeg, .png</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        st.image(uploaded_file, width=140)

    st.text_area(
        "Message",
        placeholder="Enter your question about body language...",
        height=96,
        key="message_box",
        label_visibility="collapsed",
        disabled=st.session_state.is_processing,
    )

    send = st.button(
        "Send",
        key="send_button",
        disabled=st.session_state.is_processing,
    )

    if send:
        st.session_state.error_message = ""
        query = st.session_state.message_box.strip()

        if not query:
            st.session_state.error_message = "Please enter a question before sending."
            st.rerun()

        saved_image_path = _save_uploaded_file(uploaded_file)
        if st.session_state.error_message:
            st.rerun()

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.pending_query = query
        st.session_state.pending_image_path = saved_image_path
        st.session_state.is_processing = True
        st.rerun()

    if st.session_state.is_processing:
        st.info("Processing your request...")

    if st.session_state.error_message:
        st.markdown(
            f'<div class="error-panel">{st.session_state.error_message}</div>',
            unsafe_allow_html=True,
        )


# =========================
# App
# =========================

def main() -> None:
    if st.session_state.panel_open:
        left_col, right_col = st.columns([1.05, 3.0], gap="large")
        with left_col:
            _render_left_panel()
        with right_col:
            _render_main()
    else:
        _render_main()


if __name__ == "__main__":
    main()