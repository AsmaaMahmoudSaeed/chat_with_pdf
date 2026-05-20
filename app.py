import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import tempfile
import requests
import time

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI PDF Avatar Tutor",
    layout="wide"
)

st.title("📄🤖 AI PDF Avatar Tutor")

st.write("✅ System Ready")

# =========================================
# API KEYS
# =========================================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HEYGEN_API_KEY = st.secrets["HEYGEN_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# =========================================
# GROQ CLIENT
# =========================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =========================================
# PDF EXTRACTION
# =========================================

def extract_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        t = page.extract_text()

        if t:
            text += t

    return text

# =========================================
# CHUNKING
# =========================================

def chunk_text(text, size=500):

    chunks = []

    for i in range(0, len(text), size):

        chunks.append(
            text[i:i+size]
        )

    return chunks

# =========================================
# TF-IDF SEARCH
# =========================================

def build_search_engine(chunks):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(chunks)

    return vectorizer, vectors

# =========================================
# SEARCH
# =========================================

def search(query, vectorizer, vectors, chunks):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        vectors
    ).flatten()

    top_indices = similarities.argsort()[-3:][::-1]

    results = []

    for idx in top_indices:

        results.append(chunks[idx])

    return results

# =========================================
# TEXT TO SPEECH
# =========================================

def speak(text):

    tts = gTTS(
        text=text,
        lang="ar"
    )

    file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    tts.save(file.name)

    return file.name

# =========================================
# HEYGEN VIDEO
# =========================================

def generate_heygen_video(text):

    text = text[:300]

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

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        data = response.json()

        st.write("HeyGen Response")
        st.json(data)

        if not data.get("data"):

            return None

        video_id = data["data"].get(
            "video_id"
        )

        if not video_id:

            return None

        status_url = (
            f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
        )

        for _ in range(24):

            r = requests.get(
                status_url,
                headers=headers
            )

            s = r.json()

            if not s.get("data"):

                time.sleep(5)

                continue

            status = s["data"].get(
                "status"
            )

            if status == "completed":

                return s["data"].get(
                    "video_url"
                )

            if status == "failed":

                st.error(
                    "Video generation failed"
                )

                st.json(s)

                return None

            time.sleep(5)

        return None

    except Exception as e:

        st.error(f"HeyGen Error: {e}")

        return None

# =========================================
# YOUTUBE SEARCH
# =========================================

def search_youtube(query):

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {

        "part": "snippet",

        "q": query + " شرح عربي",

        "key": YOUTUBE_API_KEY,

        "maxResults": 3,

        "type": "video",

        "relevanceLanguage": "ar"
    }

    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    videos = []

    for item in data.get("items", []):

        video_id = item["id"]["videoId"]

        title = item["snippet"]["title"]

        videos.append({

            "title": title,

            "url": f"https://www.youtube.com/watch?v={video_id}"
        })

    return videos

# =========================================
# SESSION STATE
# =========================================

if "vectorizer" not in st.session_state:

    st.session_state.vectorizer = None

if "vectors" not in st.session_state:

    st.session_state.vectors = None

if "chunks" not in st.session_state:

    st.session_state.chunks = None

# =========================================
# PDF UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type="pdf"
)

if uploaded_file:

    if st.session_state.vectorizer is None:

        with st.spinner("Processing PDF..."):

            text = extract_pdf(uploaded_file)

            chunks = chunk_text(text)

            vectorizer, vectors = build_search_engine(chunks)

            st.session_state.vectorizer = vectorizer

            st.session_state.vectors = vectors

            st.session_state.chunks = chunks

        st.success("✅ PDF Ready")

# =========================================
# TEXT QUESTION
# =========================================

question = st.text_input(
    "✍️ Ask your question"
)

# =========================================
# VOICE QUESTION
# =========================================

st.write("🎤 Voice Question")

try:

    audio = mic_recorder(

        start_prompt="🎤 Start Recording",

        stop_prompt="⛔ Stop Recording",

        key="mic"
    )

except Exception as e:

    st.error(f"Microphone Error: {e}")

    audio = None

voice_text = None

# =========================================
# SPEECH TO TEXT
# =========================================

if audio:

    audio_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    audio_file.write(audio["bytes"])

    audio_file.close()

    with open(audio_file.name, "rb") as f:

        result = client.audio.transcriptions.create(

            model="whisper-large-v3",

            file=f
        )

        voice_text = result.text

    st.info(voice_text)

# =========================================
# FINAL QUESTION
# =========================================

final_question = voice_text or question

# =========================================
# ANSWER GENERATION
# =========================================

if (
    final_question and
    st.session_state.vectorizer is not None
):

    with st.spinner("Generating Answer..."):

        retrieved_chunks = search(

            final_question,

            st.session_state.vectorizer,

            st.session_state.vectors,

            st.session_state.chunks
        )

        context = "\n".join(retrieved_chunks)

        prompt = f"""
        أجب فقط من النص التالي:

        {context}

        السؤال:
        {final_question}
        """

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[

                {
                    "role": "system",
                    "content": "أنت مدرس عربي ذكي"
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        answer = response.choices[0].message.content

    # =====================================
    # ANSWER
    # =====================================

    st.subheader("📌 Answer")

    st.write(answer)

    # =====================================
    # AUDIO
    # =====================================

    audio_path = speak(answer)

    st.audio(audio_path)

    # =====================================
    # AVATAR VIDEO
    # =====================================

    st.subheader("🎬 AI Avatar")

    with st.spinner("Generating AI video..."):

        video = generate_heygen_video(answer)

    # =====================================
    # SUCCESS
    # =====================================

    if video:

        st.success("Avatar generated ✅")

        st.video(video)

    # =====================================
    # FALLBACK
    # =====================================

    else:

        st.warning(
            "Avatar API unavailable. Fallback mode activated."
        )

        st.image(
            "https://i.imgur.com/6VBx3io.png",
            width=300
        )

        st.info(
            "Audio explanation is available below."
        )

    # =====================================
    # YOUTUBE VIDEOS
    # =====================================

    st.subheader("🎥 Arabic Educational Videos")

    try:

        videos = search_youtube(final_question)

        for v in videos:

            st.write(v["title"])

            st.video(v["url"])

    except Exception:

        st.warning(
            "Unable to load YouTube videos"
        )
