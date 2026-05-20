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

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI PDF Avatar Tutor",
    layout="wide"
)

st.title("📄🤖 AI PDF Avatar Tutor")

# =====================================
# SECRETS
# =====================================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HEYGEN_API_KEY = st.secrets["HEYGEN_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# =====================================
# GROQ CLIENT
# =====================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =====================================
# EMBEDDING MODEL
# =====================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_model()

# =====================================
# PDF EXTRACTION
# =====================================

def extract_pdf(file):
    # AVATAR VIDEO
    # =================================

    st.subheader("🎬 AI Avatar")

    with st.spinner("Generating AI video..."):

        video = generate_heygen_video(answer)

    # =================================
    # SUCCESS
    # =================================

    if video:

        st.success("Avatar generated ✅")

        st.video(video)

    # =================================
    # FALLBACK
    # =================================

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

    # =================================
    # YOUTUBE VIDEOS
    # =================================

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
