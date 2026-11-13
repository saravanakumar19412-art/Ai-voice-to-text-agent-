import os
import io
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

APP_TITLE = "AI Voice-to-Text Agent"
ASR_MODEL = "openai/whisper-large-v3"
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SUPPORTED_TYPES = ["wav", "mp3", "m4a"]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f6f4fb; color: #241a39; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #24123f 0%, #32165c 55%, #211039 100%);
    }
    [data-testid="stSidebar"] * { color: #f7f1ff !important; }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div { color: white; }
    .hero {
        background: linear-gradient(115deg, #32165c 0%, #6736a5 55%, #9562d2 100%);
        padding: 30px 34px; border-radius: 22px; color: white;
        box-shadow: 0 14px 35px rgba(58, 28, 93, .20);
        margin-bottom: 22px; position: relative; overflow: hidden;
    }
    .hero:after {
        content: ""; position: absolute; width: 250px; height: 250px;
        border: 1px solid rgba(255,255,255,.15); border-radius: 50%;
        right: -55px; top: -115px; box-shadow: 0 0 0 22px rgba(255,255,255,.04),
        0 0 0 48px rgba(255,255,255,.03);
    }
    .hero h1 { font-size: 2rem; line-height: 1.15; margin: 0 0 8px 0; font-weight: 800; }
    .hero p { margin: 0; color: #eee3ff; font-size: 1rem; }
    .hero-kicker { letter-spacing: .16em; text-transform: uppercase; font-size: .72rem;
        font-weight: 700; color: #d9c3ff; margin-bottom: 10px; }
    .panel {
        background: white; border: 1px solid #eee8f7; border-radius: 18px;
        padding: 21px 22px; box-shadow: 0 7px 24px rgba(47, 26, 75, .055);
        margin-bottom: 16px;
    }
    .panel-lilac {
        background: linear-gradient(145deg, #fff 0%, #f6f0ff 100%);
        border: 1px solid #e9ddfa; border-radius: 18px; padding: 21px 22px;
        box-shadow: 0 7px 24px rgba(47, 26, 75, .055); margin-bottom: 16px;
    }
    .section-title { color: #32165c; font-weight: 800; font-size: 1.05rem; margin-bottom: 4px; }
    .muted { color: #81768f; font-size: .88rem; }
    .metric-pill { display:inline-block; padding:5px 10px; border-radius:99px;
        background:#efe6fb; color:#603493; font-size:.76rem; font-weight:700; }
    .stButton > button {
        border-radius: 11px; border: 1px solid #e3d7f2; font-weight: 700;
        min-height: 2.65rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #5b2d91, #8250bd); border: 0; color: white;
    }
    .stTextArea textarea { border-radius: 12px; border-color: #e6def0; }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #cbb7e4; border-radius: 13px; background: #fbf9ff;
    }
    .chat-user { background:#efe7fa; border-radius:14px 14px 4px 14px; padding:13px 15px; margin:8px 0 8px 15%; }
    .chat-assistant { background:#f7f5fa; border:1px solid #eee8f5; border-radius:14px 14px 14px 4px; padding:13px 15px; margin:8px 15% 8px 0; }
    .chat-label { color:#76539c; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
    .wave { display:flex; align-items:center; gap:4px; height:34px; margin:8px 0 2px; }
    .wave span { display:block; width:4px; border-radius:6px; background:linear-gradient(180deg,#a77bda,#5a2e8e); }
    .footnote { color:#958aa2; font-size:.76rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def card_heading(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="muted">{subtitle}</div>', unsafe_allow_html=True)

def get_token() -> Optional[str]:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    try:
        token = token or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
    except Exception:
        pass
    return token.strip() if token else None

def transcribe_audio(audio_bytes: bytes, token: str, language_mode: str) -> dict:
    client = InferenceClient(model=ASR_MODEL, token=token)
    kwargs = {"audio": audio_bytes}
    # Whisper can infer the language automatically. If a language is selected,
    # pass it as a hint; do not translate the transcript.
    if language_mode != "Auto detect":
        kwargs["language"] = "tamil" if language_mode == "Tamil" else "english"
    result = client.automatic_speech_recognition(**kwargs)
    if isinstance(result, dict):
        transcript = result.get("text", "")
        detected = result.get("language") or (
            "Tamil" if language_mode == "Tamil" else
            "English" if language_mode == "English" else "Not provided by endpoint"
        )
    else:
        transcript, detected = str(result), "Not provided by endpoint"
    if not transcript.strip():
        raise RuntimeError("The speech service returned an empty transcript.")
    return {"text": transcript.strip(), "language": detected}

def make_chat_chain(token: str, temperature: float, max_tokens: int):
    prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the user's request clearly and accurately.
Use the transcript as context when relevant. If the transcript does not contain the
answer, say so rather than inventing details.

Conversation so far:
{history}

Transcript:
{transcript}

User:
{question}

Assistant:"""
    )
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        task="text-generation",
        huggingfacehub_api_token=token,
        temperature=temperature,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        repetition_penalty=1.03,
    )
    return prompt | endpoint | StrOutputParser()

def render_copy_button(text: str, element_id: str) -> None:
    """Render a real browser-side clipboard button (with a legacy fallback)."""
    import json
    safe_text = json.dumps(text)
    components.html(
        f"""<button id="{element_id}" style="width:100%;height:42px;border:1px solid #e3d7f2;"
        f"border-radius:11px;background:#fff;color:#4f2b79;font-weight:700;cursor:pointer;">📋 Copy Text</button>
        <script>
        const btn = document.getElementById({json.dumps(element_id)});
        const value = {safe_text};
        btn.addEventListener('click', async () => {{
          try {{
            if (navigator.clipboard && window.isSecureContext) {{
              await navigator.clipboard.writeText(value);
            }} else {{
              const area = document.createElement('textarea');
              area.value = value; document.body.appendChild(area);
              area.select(); document.execCommand('copy'); area.remove();
            }}
            btn.textContent = '✓ Copied';
          }} catch (err) {{ btn.textContent = 'Copy unavailable'; }}
        }});
        </script>""",
        height=50,
    )

def reset_audio_state():
    for key in ("transcript", "detected_language", "audio_bytes", "audio_name", "audio_source"):
        st.session_state.pop(key, None)

# ---------- Session state ----------
defaults = {
    "messages": [],
    "transcript": "",
    "detected_language": "",
    "audio_bytes": None,
    "audio_name": "",
    "audio_source": "",
    "last_response": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎙️ Voice Agent")
    st.caption("WORKSPACE SETTINGS")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
                            help="Lower values produce more focused answers.")
    max_tokens = st.slider("Max Tokens", 64, 1024, 512, 32)
    st.markdown("---")
    st.markdown("**Selected AI models**")
    st.markdown("`Whisper Large v3`")
    st.markdown("`Llama 3.1 8B Instruct`")
    st.caption("Hugging Face Inference")
    st.markdown("---")
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response = ""
        st.rerun()
    st.markdown('<div class="footnote">Your API token stays in your local environment.</div>',
                unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Your intelligent audio workspace</div>
      <h1>🎙️ AI Voice-to-Text Agent</h1>
      <p>Speak&nbsp; · &nbsp;Transcribe&nbsp; · &nbsp;Ask</p>
      <div class="wave" aria-label="Decorative audio waveform">
        <span style="height:10px"></span><span style="height:20px"></span>
        <span style="height:29px"></span><span style="height:15px"></span>
        <span style="height:25px"></span><span style="height:33px"></span>
        <span style="height:18px"></span><span style="height:27px"></span>
        <span style="height:12px"></span><span style="height:22px"></span>
        <span style="height:31px"></span><span style="height:15px"></span>
        <span style="height:25px"></span><span style="height:10px"></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

token = get_token()
if not token:
    st.warning(
        "Hugging Face API token not found. Add "
        "`HUGGINGFACEHUB_API_TOKEN=hf_...` to your `.env` file, then restart Streamlit."
    )

# ---------- Main two-column workspace ----------
left, right = st.columns([1.03, 0.97], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    card_heading("🎧  Voice Input", "Record, upload, or use a local sample audio file.")
    input_mode = st.radio(
        "Choose audio source",
        ["Record Voice", "Upload Audio", "Sample Audio"],
        horizontal=True,
        label_visibility="collapsed",
    )
    audio_bytes = None
    audio_name = ""

    if input_mode == "Record Voice":
        st.caption("Use the recorder below. Select Stop to finish and keep the recording.")
        try:
            from streamlit_mic_recorder import mic_recorder
            recording = mic_recorder(
                start_prompt="● Start recording",
                stop_prompt="■ Stop recording",
                just_once=True,
                use_container_width=True,
                key="voice_recorder",
            )
            if recording and recording.get("bytes"):
                audio_bytes = recording["bytes"]
                audio_name = "microphone-recording.wav"
                st.audio(audio_bytes, format="audio/wav")
                st.success("Recording captured. You can now transcribe it.")
        except ImportError:
            st.error("Microphone recording dependency is unavailable. Install requirements.txt and restart.")
    elif input_mode == "Upload Audio":
        uploaded = st.file_uploader("Choose a WAV, MP3, or M4A file", type=SUPPORTED_TYPES)
        if uploaded:
            audio_bytes = uploaded.getvalue()
            audio_name = uploaded.name
            st.audio(audio_bytes)
    else:
        sample_dir = Path(__file__).parent / "samples"
        sample_files = []
        for ext in SUPPORTED_TYPES:
            sample_files.extend(sample_dir.glob(f"*.{ext}"))
        if sample_files:
            selected = st.selectbox("Select a sample audio file", sample_files, format_func=lambda p: p.name)
            if selected:
                audio_bytes = selected.read_bytes()
                audio_name = selected.name
                st.audio(audio_bytes)
        else:
            st.info("No sample audio is bundled. Add a real WAV, MP3, or M4A file to the `samples` folder to enable this option.")

    language_mode = st.selectbox("Speech language", ["Auto detect", "Tamil", "English"])
    if audio_bytes:
        st.markdown('<span class="metric-pill">Audio ready</span>', unsafe_allow_html=True)
        st.caption(f"Selected: {audio_name}")
        st.session_state.audio_bytes = audio_bytes
        st.session_state.audio_name = audio_name
        st.session_state.audio_source = input_mode
    else:
        st.caption("Waiting for audio input.")

    transcribe_col, reset_col = st.columns([1.4, 1])
    with transcribe_col:
        transcribe_clicked = st.button("✨  Transcribe Audio", type="primary", use_container_width=True,
                                       disabled=not bool(audio_bytes) or not bool(token))
    with reset_col:
        if st.button("↻  Reset Audio", use_container_width=True):
            reset_audio_state()
            st.rerun()

    if transcribe_clicked:
        try:
            with st.spinner("Connecting to Whisper Large v3 and transcribing audio…"):
                result = transcribe_audio(audio_bytes, token, language_mode)
            st.session_state.transcript = result["text"]
            st.session_state.detected_language = str(result["language"])
            st.session_state.audio_bytes = audio_bytes
            st.session_state.audio_name = audio_name
            st.success("Transcription completed.")
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            st.caption("Check your token, model access, audio format, and Hugging Face Inference availability.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-lilac">', unsafe_allow_html=True)
    card_heading("💬  Quick actions", "Start with a useful question about the current transcript.")
    quick_cols = st.columns(3)
    quick_prompts = [
        ("Summarize", "Summarize the transcript in concise bullet points."),
        ("Key points", "Extract the key points and important details."),
        ("Explain", "Explain the transcript in simple language."),
    ]
    for col, (label, prompt_text) in zip(quick_cols, quick_prompts):
        with col:
            if st.button(label, use_container_width=True, disabled=not bool(st.session_state.transcript)):
                st.session_state.pending_question = prompt_text
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    card_heading("📝  Transcription Result", "Review and edit the recognized speech.")
    if st.session_state.transcript:
        st.markdown(
            f'<span class="metric-pill">Language: {st.session_state.detected_language or "Unknown"}</span>',
            unsafe_allow_html=True,
        )
    edited_transcript = st.text_area(
        "Transcribed text",
        value=st.session_state.transcript,
        height=280,
        placeholder="Your transcription will appear here…",
        key="transcript_editor",
    )
    # Keep edits as the authoritative context for subsequent chat and downloads.
    st.session_state.transcript = edited_transcript
    copy_col, download_col = st.columns(2)
    with copy_col:
        if edited_transcript:
            render_copy_button(edited_transcript, "copy-transcript-button")
        else:
            st.button("📋  Copy Text", use_container_width=True, disabled=True)
    with download_col:
        st.download_button(
            "⬇️  Download TXT",
            data=edited_transcript.encode("utf-8"),
            file_name="transcription.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not bool(edited_transcript),
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Full-width assistant ----------
st.markdown('<div class="panel">', unsafe_allow_html=True)
card_heading("🤖  AI Assistant", "Ask questions about your transcript or start a normal conversation.")
if not st.session_state.messages:
    st.markdown('<div class="muted">Your conversation will appear here. Transcribe audio first to ask transcript-specific questions.</div>',
                unsafe_allow_html=True)

for message in st.session_state.messages:
    css_class = "chat-user" if message["role"] == "user" else "chat-assistant"
    label = "YOU" if message["role"] == "user" else "AI ASSISTANT"
    st.markdown(
        f'<div class="{css_class}"><div class="chat-label">{label}</div>'
        f'<div>{message["content"]}</div></div>',
        unsafe_allow_html=True,
    )

pending = st.session_state.pop("pending_question", "")
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Message", value=pending, placeholder="Ask anything about the transcript…")
    send = st.form_submit_button("Send message  ➜", type="primary", use_container_width=True)

if send and question.strip():
    if not token:
        st.error("Add your Hugging Face API token to .env before sending a message.")
    else:
        st.session_state.messages.append({"role": "user", "content": question.strip()})
        history = "\n".join(
            f'{m["role"].capitalize()}: {m["content"]}'
            for m in st.session_state.messages[:-1][-10:]
        )
        try:
            with st.spinner("Llama 3.1 is preparing a response…"):
                chain = make_chat_chain(token, temperature, max_tokens)
                answer = chain.invoke({
                    "history": history or "No previous conversation.",
                    "transcript": st.session_state.transcript or "No transcript supplied.",
                    "question": question.strip(),
                })
            st.session_state.messages.append({"role": "assistant", "content": answer.strip()})
            st.session_state.last_response = answer.strip()
            st.rerun()
        except Exception as exc:
            # Keep the user's message in history, but do not fabricate an answer.
            st.error(f"AI response failed: {exc}")
            st.caption("Verify Hugging Face token permissions, model availability, and inference quota.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footnote" style="text-align:center;padding:8px 0 20px;">'
    'Powered by Hugging Face Whisper Large v3 · Llama 3.1 8B Instruct · LangChain'
    '</div>',
    unsafe_allow_html=True,
)
