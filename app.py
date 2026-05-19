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
import os

# =============================
# إعداد الصفحة
# =============================

st.set_page_config(
    page_title="Smart Arabic PDF Tutor",
    layout="wide"
)

st.title("📄🎤🤖 Smart Arabic PDF Tutor")

# =============================
# API KEYS
# =============================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =============================
# تحميل نموذج Embedding
# =============================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# =============================
# قراءة PDF
# =============================

def extract_text_from_pdf(pdf_file):

    pdf_reader = PdfReader(pdf_file)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

# =============================
# تقسيم النص
# =============================

def split_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(
            text[i:i + chunk_size]
        )

    return chunks

# =============================
# إنشاء FAISS
# =============================

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index

# =============================
# البحث
# =============================

def search(query, index, chunks, k=3):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results

# =============================
# تحويل النص إلى صوت
# =============================

def text_to_speech(text):

    tts = gTTS(
        text=text,
        lang='ar'
    )

    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    tts.save(temp_audio.name)

    return temp_audio.name

# =============================
# البحث في يوتيوب
# =============================

def search_youtube_videos(query, max_results=3):

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": query + " شرح عربي",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
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

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        videos.append({
            "title": title,
            "url": video_url
        })

    return videos

# =============================
# رفع PDF
# =============================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    with st.spinner("معالجة الملف ..."):

        pdf_text = extract_text_from_pdf(
            uploaded_file
        )

        chunks = split_text(pdf_text)

        index = create_faiss_index(
            chunks
        )

    st.success("تمت المعالجة ✅")

    # =========================
    # سؤال نصي
    # =========================

    text_question = st.text_input(
        "✍️ اكتب سؤالك"
    )

    # =========================
    # سؤال صوتي
    # =========================

    st.write("🎤 أو اسأل بالصوت")

    audio = mic_recorder(
        start_prompt="ابدأ التسجيل",
        stop_prompt="إيقاف التسجيل",
        key="mic"
    )

    voice_question = None

    if audio:

        audio_bytes = audio["bytes"]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_audio_file:

            temp_audio_file.write(audio_bytes)

            temp_audio_path = temp_audio_file.name

        with open(temp_audio_path, "rb") as audio_file:

            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file
            )

        voice_question = transcription.text

        st.info(voice_question)

    question = voice_question or text_question

    if question:

        with st.spinner("جاري التوليد ..."):

            retrieved_chunks = search(
                question,
                index,
                chunks
            )

            context = "\n".join(
                retrieved_chunks
            )

            prompt = f"""
            أجب فقط باستخدام النص التالي.

            إذا لم توجد الإجابة قل:
            المعلومة غير موجودة في الملف

            النص:
            {context}

            السؤال:
            {question}
            """

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "أنت مدرس عربي ذكي."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2
            )

            answer = response.choices[0].message.content

        st.subheader("📌 الإجابة")
        st.write(answer)

        # =========================
        # صوت الإجابة
        # =========================

        audio_path = text_to_speech(
            answer
        )

        st.audio(audio_path)

        # =========================
        # فيديوهات يوتيوب
        # =========================

        st.subheader("🎥 فيديوهات مقترحة")

        videos = search_youtube_videos(
            question + " " + answer[:80]
        )

        for video in videos:

            st.write(video["title"])

            st.video(video["url"])

        # =========================
        # النصوص المستخدمة
        # =========================

        with st.expander("النصوص المستخدمة"):

            for chunk in retrieved_chunks:
                st.info(chunk)
