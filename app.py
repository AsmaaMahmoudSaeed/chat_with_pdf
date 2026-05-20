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

# =========================================
# إعداد الصفحة
# =========================================

st.set_page_config(
    page_title="AI Arabic PDF Avatar Tutor",
    layout="wide"
)

st.title("📄🎤🤖 AI Arabic PDF Avatar Tutor")

# =========================================
# API KEYS
# =========================================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

HEYGEN_API_KEY = st.secrets["HEYGEN_API_KEY"]

# =========================================
# Groq Client
# =========================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =========================================
# Embedding Model
# =========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# =========================================
# استخراج النص من PDF
# =========================================

def extract_text_from_pdf(pdf_file):

    pdf_reader = PdfReader(pdf_file)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text

    return text

# =========================================
# تقسيم النص
# =========================================

def split_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(
            text[i:i + chunk_size]
        )

    return chunks

# =========================================
# إنشاء FAISS
# =========================================

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

# =========================================
# البحث
# =========================================

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

# =========================================
# تحويل النص إلى صوت
# =========================================

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

# =========================================
# البحث في يوتيوب
# =========================================

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

        video_url = (
            f"https://www.youtube.com/watch?v={video_id}"
        )

        videos.append({
            "title": title,
            "url": video_url
        })

    return videos

# =========================================
# إنشاء فيديو HeyGen
# =========================================

def generate_heygen_video(answer_text):

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

                    "input_text": answer_text,

                    "voice_id": "ar-SA-HamedNeural"
                },

                "background": {

                    "type": "color",

                    "value": "#f6f6f6"
                }
            }
        ]
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    data = response.json()

    if "data" not in data:

        return None

    video_id = data["data"]["video_id"]

    # =====================================
    # انتظار تجهيز الفيديو
    # =====================================

    status_url = (
        f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    )

    for _ in range(30):

        status_response = requests.get(
            status_url,
            headers=headers
        )

        status_data = status_response.json()

        if (
            "data" in status_data and
            status_data["data"]["status"] == "completed"
        ):

            return status_data["data"]["video_url"]

        time.sleep(5)

    return None

# =========================================
# رفع PDF
# =========================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    with st.spinner("جاري معالجة PDF ..."):

        pdf_text = extract_text_from_pdf(
            uploaded_file
        )

        chunks = split_text(pdf_text)

        index = create_faiss_index(
            chunks
        )

    st.success("تمت معالجة الملف بنجاح ✅")

    # =====================================
    # سؤال نصي
    # =====================================

    text_question = st.text_input(
        "✍️ اكتب سؤالك"
    )

    # =====================================
    # سؤال صوتي
    # =====================================

    st.write("🎤 أو اسأل بالصوت")

    try:

        audio = mic_recorder(
            start_prompt="ابدأ التسجيل",
            stop_prompt="إيقاف التسجيل",
            key="mic"
        )

    except Exception as e:

        st.error(f"Microphone Error: {e}")

        audio = None

    voice_question = None

    # =====================================
    # تحويل الصوت إلى نص
    # =====================================

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

    # =====================================
    # الإجابة
    # =====================================

    if question:

        with st.spinner("جاري توليد الإجابة ..."):

            retrieved_chunks = search(
                question,
                index,
                chunks
            )

            context = "\n\n".join(
                retrieved_chunks
            )

            prompt = f"""
            أجب فقط باستخدام المعلومات الموجودة في النص التالي.

            إذا لم تجد الإجابة داخل النص قل:
            "المعلومة غير موجودة في الملف"

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
                        "content": (
                            "أنت مدرس عربي ذكي يشرح"
                            " بطريقة سهلة."
                        )
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
        # عرض الإجابة النصية
        # =====================================

        st.subheader("📌 الإجابة")

        st.write(answer)

        # =====================================
        # الإجابة الصوتية
        # =====================================

        audio_path = text_to_speech(
            answer
        )

        st.audio(audio_path)

        # =====================================
        # فيديو Avatar
        # =====================================

        st.subheader("🎥 AI Avatar Video")

        with st.spinner(
            "جاري إنشاء فيديو الذكاء الاصطناعي..."
        ):

            video_url = generate_heygen_video(
                answer
            )

        if video_url:

            st.video(video_url)

        else:

            st.warning(
                "تعذر إنشاء الفيديو حالياً"
            )

        # =====================================
        # فيديوهات يوتيوب
        # =====================================

        st.subheader("🎥 فيديوهات تعليمية مقترحة")

        try:

            videos = search_youtube_videos(
                question + " " + answer[:80]
            )

            for video in videos:

                st.write(video["title"])

                st.video(video["url"])

        except Exception as e:

            st.warning(
                "تعذر تحميل فيديوهات يوتيوب"
            )

        # =====================================
        # النصوص المستخدمة
        # =====================================

        with st.expander(
            "📄 النصوص المستخدمة"
        ):

            for chunk in retrieved_chunks:

                st.info(chunk)
