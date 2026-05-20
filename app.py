import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from gtts import gTTS
import tempfile
import requests
import time
import base64

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="AI PDF Tutor (D-ID)", layout="wide")

st.title("📄🤖 AI PDF Tutor (D-ID Avatar)")

st.write("Powered by D-ID Avatar System")

# =========================
# SECRETS
# =========================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DID_API_KEY = st.secrets["DID_API_KEY"]
DID_API_URL = "https://api.d-id.com/talks"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

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
# SEARCH (TF-IDF)
# =========================

def build_index(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors


def search(query, vectorizer, vectors, chunks):
    qv = vectorizer.transform([query])
    scores = cosine_similarity(qv, vectors).flatten()
    idx = scores.argsort()[-3:][::-1]
    return [chunks[i] for i in idx]

# =========================
# TTS fallback
# =========================

def speak(text):
    tts = gTTS(text=text, lang="ar")
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# =========================
# D-ID AVATAR VIDEO
# =========================

def generate_did_video(text):

    headers = {
        "Authorization": f"Basic {DID_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "source_url": "https://create-images-results.d-id.com/DefaultPresenters/Noelle_f/image.png",
        "script": {
            "type": "text",
            "input": text[:500],
            "provider": {
                "type": "microsoft",
                "voice_id": "ar-EG-SalmaNeural"
            }
        }
    }

    try:
        r = requests.post(DID_API_URL, json=payload, headers=headers)
        data = r.json()

        st.write("D-ID Response")
        st.json(data)

        talk_id = data.get("id")
        if not talk_id:
            return None

        # polling
        status_url = f"https://api.d-id.com/talks/{talk_id}"

        for _ in range(30):
            res = requests.get(status_url, headers=headers)
            s = res.json()

            if s.get("status") == "done":
                return s["result_url"]

            if s.get("status") == "error":
                return None

            time.sleep(3)

        return None

    except Exception as e:
        st.error(f"D-ID Error: {e}")
        return None

# =========================
# YOUTUBE
# =========================

def search_youtube(query):
    key = st.secrets["YOUTUBE_API_KEY"]

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": query + " شرح عربي",
        "key": key,
        "maxResults": 3,
        "type": "video",
        "relevanceLanguage": "ar"
    }

    r = requests.get(url, params=params)
    data = r.json()

    videos = []

    for item in data.get("items", []):
        vid = item["id"]["videoId"]
        title = item["snippet"]["title"]
        videos.append({
            "title": title,
            "url": f"https://www.youtube.com/watch?v={vid}"
        })

    return videos

# =========================
# SESSION
# =========================

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
    st.session_state.vectors = None
    st.session_state.chunks = None

# =========================
# UI
# =========================

file = st.file_uploader("Upload PDF", type="pdf")

if file and st.session_state.vectorizer is None:
    text = extract_pdf(file)
    chunks = chunk_text(text)

    vectorizer, vectors = build_index(chunks)

    st.session_state.vectorizer = vectorizer
    st.session_state.vectors = vectors
    st.session_state.chunks = chunks

    st.success("PDF Loaded ✅")

question = st.text_input("Ask your question")

# =========================
# MAIN LOGIC
# =========================

if question and st.session_state.vectorizer:

    context = search(
        question,
        st.session_state.vectorizer,
        st.session_state.vectors,
        st.session_state.chunks
    )

    prompt = f"""
    أجب من النص فقط:

    {context}

    السؤال: {question}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "أنت مدرس عربي"},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    st.subheader("📌 Answer")
    st.write(answer)

    # AUDIO
    audio = speak(answer)
    st.audio(audio)

    # =========================
    # D-ID VIDEO
    # =========================

    st.subheader("🎬 AI Avatar (D-ID)")

    with st.spinner("Generating video..."):
        video = generate_did_video(answer)

    if video:
        st.success("Video Ready ✅")
        st.video(video)
    else:
        st.warning("Fallback Mode (Audio only)")
        st.info("D-ID video generation failed")

    # =========================
    # YOUTUBE
    # =========================

    st.subheader("🎥 YouTube Explanation")

    try:
        vids = search_youtube(question)

        for v in vids:
            st.write(v["title"])
            st.video(v["url"])

    except:
        st.warning("YouTube error")
