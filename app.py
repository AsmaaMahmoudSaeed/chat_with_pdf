import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import tempfile
import faiss
import numpy as np
import requests
import time

# =========================
# إعداد التطبيق
# =========================

st.set_page_config(
    page_title="AI PDF Tutor + Avatar",
    layout="wide"
)

st.title("📄🤖🎤 AI PDF Tutor + HeyGen Avatar")

# =========================
# API KEYS
# =========================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HEYGEN_API_KEY = st.secrets["HEYGEN_API_KEY"]

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =========================
# Embedding Model
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# PDF
# =========================

def extract_pdf(file):
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t
    return text


def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# FAISS
# =========================

def build_index(chunks):
    emb = model.encode(chunks)
    emb = np.array(emb, dtype=np.float32)

    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    return index


def search(query, index, chunks):
    q_emb = model.encode([query])
    q_emb = np.array(q_emb, dtype=np.float32)

    _, idx = index.search(q_emb, 3)

    return [chunks[i] for i in idx[0]]

# =========================
# TTS
# =========================

def speak(text):
    tts = gTTS(text, lang="ar")
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# =========================
# 🔥 HEYGEN FIXED VERSION
# =========================

def generate_heygen_video(text):

    url = "https://api.heygen.com/v2/video/generate"

    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": "Daisy-inskirt-20220818"
                },
                "voice": {
                    "type": "text",
                    "voice_id": "2d5b0e6cf36f460aa7fc47e3eee4ba54",
                    "input_text": text
                },
                "background": {
                    "type": "color",
                    "value": "#ffffff"
                }
            }
        ]
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        data = res.json()

        st.write("HeyGen Response:")
        st.json(data)

        if not data.get("data"):
            st.error(data.get("error", {}).get("message", "HeyGen error"))
            return None

        video_id = data["data"].get("video_id")
        if not video_id:
            st.error("No video_id returned")
            return None

        status_url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"

        for _ in range(30):
            r = requests.get(status_url, headers=headers)
            s = r.json()

            if not s.get("data"):
                time.sleep(5)
                continue

            status = s["data"].get("status")

            if status == "completed":
                return s["data"].get("video_url")

            if status == "failed":
                st.error("Video generation failed")
                return None

            time.sleep(5)

        st.warning("Timeout waiting video")
        return None

    except Exception as e:
        st.error(f"HeyGen error: {e}")
        return None

# =========================
# Upload PDF
# =========================

file = st.file_uploader("Upload PDF", type="pdf")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

if file:

    if st.session_state.index is None:

        text = extract_pdf(file)
        chunks = chunk_text(text)
        index = build_index(chunks)

        st.session_state.index = index
        st.session_state.chunks = chunks

        st.success("PDF Loaded ✅")

# =========================
# Input
# =========================

question = st.text_input("Ask your question (text or voice)")

audio = mic_recorder(
    start_prompt="🎤 Start",
    stop_prompt="⛔ Stop",
    key="mic"
)

voice_text = None

if audio:
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    path.write(audio["bytes"])
    path.close()

    with open(path.name, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f
        )
        voice_text = result.text

    st.info(voice_text)

final_q = voice_text or question

# =========================
# Answer
# =========================

if final_q and st.session_state.index:

    chunks = search(final_q, st.session_state.index, st.session_state.chunks)

    context = "\n".join(chunks)

    prompt = f"""
    أجب فقط من النص التالي:

    {context}

    السؤال:
    {final_q}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "أنت مدرس عربي مفيد"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    st.subheader("📌 Answer")
    st.write(answer)

    # صوت
    audio_file = speak(answer)
    st.audio(audio_file)

    # 🎥 HeyGen Video
    st.subheader("🎬 Avatar Video")

    video = generate_heygen_video(answer)

    if video:
        st.video(video)
    else:
        st.warning("No video generated")
