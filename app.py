import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from gtts import gTTS
import tempfile
import requests
import time

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="AI Smart Teacher",
    layout="wide"
)

st.title("📘🤖 AI Smart Teacher (Advanced RAG + Avatar)")

st.write("مدرس ذكي يعتمد على PDF + صوت + فيديو Avatar")

# =========================
# KEYS
# =========================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DID_API_KEY = st.secrets["DID_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =========================
# EMBEDDING MODEL
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

model = load_model()

# =========================
# PDF FUNCTIONS
# =========================

def extract_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# EMBEDDINGS
# =========================

def build_embeddings(chunks):
    embeddings = model.encode(chunks)
    return np.array(embeddings, dtype=np.float32)

# =========================
# SEARCH
# =========================

def search(query, chunks, embeddings):

    q_emb = model.encode([query])

    scores = cosine_similarity(q_emb, embeddings).flatten()

    top_idx = scores.argsort()[-4:][::-1]

    context = " ".join([chunks[i] for i in top_idx])

    return context

# =========================
# TEXT TO SPEECH
# =========================

def speak(text):
    tts = gTTS(text=text, lang="ar")
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# =========================
# D-ID AVATAR
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
        r = requests.post(
            "https://api.d-id.com/talks",
            json=payload,
            headers=headers
        )

        data = r.json()

        talk_id = data.get("id")
        if not talk_id:
            return None

        for _ in range(30):
            res = requests.get(
                f"https://api.d-id.com/talks/{talk_id}",
                headers=headers
            )

            s = res.json()

            if s.get("status") == "done":
                return s["result_url"]

            if s.get("status") == "error":
                return None

            time.sleep(3)

        return None

    except:
        return None

# =========================
# YOUTUBE
# =========================

def youtube(query):

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": query + " شرح عربي",
        "key": YOUTUBE_API_KEY,
        "maxResults": 3,
        "type": "video"
    }

    r = requests.get(url, params=params)
    data = r.json()

    vids = []

    for i in data.get("items", []):
        vid = i["id"]["videoId"]
        title = i["snippet"]["title"]

        vids.append({
            "title": title,
            "url": f"https://www.youtube.com/watch?v={vid}"
        })

    return vids

# =========================
# SESSION STATE
# =========================

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.embeddings = None

# =========================
# UPLOAD PDF
# =========================

file = st.file_uploader("📄 Upload PDF", type="pdf")

if file:

    text = extract_pdf(file)

    chunks = chunk_text(text)

    embeddings = build_embeddings(chunks)

    st.session_state.chunks = chunks

    st.session_state.embeddings = embeddings

    st.success("PDF جاهز ✅")

# =========================
# QUESTION
# =========================

q = st.text_input("✍️ اسأل سؤال")

# =========================
# MAIN LOGIC
# =========================

if q and st.session_state.chunks is not None:

    context = search(
        q,
        st.session_state.chunks,
        st.session_state.embeddings
    )

    prompt = f"""
أنت مدرس ذكي.
أجب فقط من النص التالي.

إذا لم تجد الإجابة قل: "غير موجود في المحتوى"

النص:
{context}

السؤال:
{q}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "مدرس عربي دقيق"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    # =========================
    # ANSWER
    # =========================

    st.subheader("📌 الإجابة")
    st.write(answer)

    # =========================
    # AUDIO
    # =========================

    audio = speak(answer)
    st.audio(audio)

    # =========================
    # AVATAR VIDEO (CUSTOM SIZE)
    # =========================

    st.subheader("🎬 شرح بالفيديو (Avatar)")

    video_url = generate_did_video(answer)

    if video_url:

        st.success("تم إنشاء الفيديو بنجاح ✅")

        st.markdown(
            f"""
            <div style="display:flex; justify-content:center;">
                <div style="border-radius:15px; overflow:hidden; box-shadow:0px 4px 20px rgba(0,0,0,0.2);">
                    <video width="420" height="260" controls>
                        <source src="{video_url}" type="video/mp4">
                    </video>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning("تعذر إنشاء فيديو Avatar")

        st.info("سيتم عرض الشرح الصوتي فقط")

    # =========================
    # YOUTUBE
    # =========================

    st.subheader("🎥 فيديوهات شرح إضافية")

    vids = youtube(q)

    for v in vids:
        st.write(v["title"])
        st.video(v["url"])
