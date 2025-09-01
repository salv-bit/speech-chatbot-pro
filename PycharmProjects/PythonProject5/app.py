# app.py — v2 Improved Speech Recognition App
import os
import time
import io
import re
import streamlit as st
import speech_recognition as sr

from chatbot import SimpleChatbot

st.set_page_config(page_title="Speech Chatbot Pro", page_icon="🗣️", layout="centered")
st.title("🗣️ Speech-Enabled Chatbot — Pro Features")
st.caption("Text & voice, multi-API, language picker, pause/resume, and transcript saving.")

# -----------------------------
# Sidebar: corpus and settings
# -----------------------------
st.sidebar.header("Knowledge Base")
corpus_file = st.sidebar.file_uploader("Upload a .txt corpus (optional)", type=["txt"])

if corpus_file:
    corpus_text = corpus_file.read().decode("utf-8", errors="ignore")
else:
    try:
        with open("corpus.txt", "r", encoding="utf-8") as f:
            corpus_text = f.read()
    except FileNotFoundError:
        corpus_text = "Q: What are your hours?\nA: We are open Monday to Friday, 9am–5pm."

bot = SimpleChatbot(corpus_text)

# Recognizer instance (reuse between runs)
if "recognizer" not in st.session_state:
    st.session_state.recognizer = sr.Recognizer()
r = st.session_state.recognizer

# -----------------------------
# API selection and language
# -----------------------------
st.sidebar.subheader("Speech Recognition Settings")

# Check if pocketsphinx is installed
try:
    import pocketsphinx  # noqa: F401
    sphinx_available = True
except Exception:
    sphinx_available = False

api_options = ["Google Web Speech"]
if sphinx_available:
    api_options.append("Offline Sphinx")

engine = st.sidebar.selectbox("Recognition API", api_options, index=0,
                              help="Google uses the web (needs internet). Sphinx is offline if installed.")

# Language codes mapping (extend as needed)
LANG_CHOICES = {
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "French": "fr-FR",
    "Arabic": "ar-SA",
    "Spanish": "es-ES",
    "German": "de-DE",
    "Italian": "it-IT",
    "Portuguese": "pt-PT",
    "Yoruba": "yo-NG",
    "Igbo": "ig-NG",
    "Hausa": "ha-NG",
}

lang_label = st.sidebar.selectbox("Spoken Language", list(LANG_CHOICES.keys()), index=0)
lang_code = LANG_CHOICES[lang_label]

noise_dur = st.sidebar.slider("Ambient noise adjust (sec)", 0.0, 2.0, 0.6, 0.1)
chunk_sec = st.sidebar.slider("Chunk length (sec)", 3, 15, 6, 1, help="Max seconds per listen chunk")

# -----------------------------
# Conversation history (chat UI)
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list[(role, text)]

# Transcript buffer (raw speech to text)
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# Recording state machine
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "is_paused" not in st.session_state:
    st.session_state.is_paused = False


def render_history():
    for role, msg in st.session_state.history[-12:]:
        st.chat_message("user" if role == "user" else "assistant").write(msg)


st.subheader("Chat")
mode = st.radio("Choose input mode:", ["Text", "Voice (microphone)"])
render_history()

# -----------------------------
# Helper: robust transcription
# -----------------------------
def transcribe_chunk(engine_name: str, language: str, timeout_s: int, phrase_limit_s: int) -> str:
    """Capture a short chunk and return text. Raises informative exceptions."""
    with sr.Microphone() as source:
        try:
            r.adjust_for_ambient_noise(source, duration=noise_dur)
            audio = r.listen(source, timeout=timeout_s, phrase_time_limit=phrase_limit_s)
        except sr.WaitTimeoutError:
            raise RuntimeError("No speech detected before timeout. Try speaking sooner or increase timeout.")
        except OSError as e:
            raise RuntimeError(f"Microphone error: {e}. Check that a mic is connected and allowed.")

    # Recognition backends
    try:
        if engine_name == "Google Web Speech":
            # Google supports language code
            return r.recognize_google(audio, language=language)
        elif engine_name == "Offline Sphinx":
            # Sphinx ignores language codes unless models are installed; still attempt
            return r.recognize_sphinx(audio)
        else:
            raise RuntimeError("Unknown engine selected.")
    except sr.UnknownValueError:
        raise RuntimeError("Audio was not clear enough to understand. Please speak clearly and try again.")
    except sr.RequestError as e:
        raise RuntimeError(f"Speech service request failed: {e}")

# -----------------------------
# TEXT MODE
# -----------------------------
if mode == "Text":
    prompt = st.chat_input("Type your message…")
    if prompt:
        st.session_state.history.append(("user", prompt))
        reply = bot.reply(prompt)
        st.session_state.history.append(("assistant", reply))
        st.rerun()

# -----------------------------
# VOICE MODE
# -----------------------------
else:
    col1, col2, col3, col4 = st.columns(4)
    start = col1.button("▶️ Start", type="primary", disabled=st.session_state.is_listening and not st.session_state.is_paused)
    pause = col2.button("⏸️ Pause", disabled=not st.session_state.is_listening or st.session_state.is_paused)
    resume = col3.button("⏯️ Resume", disabled=not st.session_state.is_paused)
    stop = col4.button("⏹️ Stop", disabled=not st.session_state.is_listening)

    if engine == "Offline Sphinx" and not sphinx_available:
        st.warning("Offline Sphinx selected but `pocketsphinx` is not installed. Falling back to Google.")
        engine = "Google Web Speech"

    # Handle state transitions
    if start:
        st.session_state.is_listening = True
        st.session_state.is_paused = False

    if pause:
        st.session_state.is_paused = True

    if resume:
        st.session_state.is_paused = False

    if stop:
        st.session_state.is_listening = False
        st.session_state.is_paused = False

    # Display status
    if st.session_state.is_listening and not st.session_state.is_paused:
        st.info(f"Listening… Engine: {engine} | Language: {lang_code}")
    elif st.session_state.is_listening and st.session_state.is_paused:
        st.warning("Paused. Click Resume to continue listening.")
    else:
        st.caption("Click Start to begin listening.")

    # If actively listening (not paused), capture one chunk per run and append
    if st.session_state.is_listening and not st.session_state.is_paused:
        try:
            text = transcribe_chunk(engine, lang_code, timeout_s=8, phrase_limit_s=int(chunk_sec))
            if text:
                st.success("Heard: " + text)
                # Add to transcript buffer
                if st.session_state.transcript:
                    st.session_state.transcript += " " + text
                else:
                    st.session_state.transcript = text
                # Also run the chatbot on this chunk
                st.session_state.history.append(("user", text))
                reply = bot.reply(text)
                st.session_state.history.append(("assistant", reply))
                # Rerun to keep capturing next chunk automatically
                st.rerun()
        except Exception as e:
            st.error(str(e))

# -----------------------------
# Transcript panel + Save / Download
# -----------------------------
st.subheader("Transcript")
st.text_area("Live transcript", value=st.session_state.transcript, height=140, key="transcript_area")

colA, colB = st.columns([1, 1])

with colA:
    filename = st.text_input("Save as filename", value="transcript.txt")
    if st.button("💾 Save to file"):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(st.session_state.transcript)
            st.success(f"Saved to {filename}")
        except Exception as e:
            st.error(f"Could not save file: {e}")

with colB:
    st.download_button(
        label="⬇️ Download transcript",
        data=st.session_state.transcript.encode("utf-8"),
        file_name="transcript.txt",
        mime="text/plain",
        disabled=not bool(st.session_state.transcript)
    )
