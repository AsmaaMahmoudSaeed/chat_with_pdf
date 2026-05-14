import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import faiss
import numpy as np

# =========================================
# إعداد الصفحة
# =========================================

st.set_page_config(
    page_title="المدرب الذكي",
    layout="wide"
)

st.title("📄🤖 المدرب الذكي")
st.write("ارفع ملف وأسأل غن أي معلومه بداخله")

# =========================================
# إعداد Groq API
# =========================================

GROQ_API_KEY = "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"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =========================================
# تحميل نموذج Embeddings
# =========================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model

embedding_model = load_embedding_model()

# =========================================
# رفع ملف PDF
# =========================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

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

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    return chunks

# =========================================
# إنشاء FAISS Index
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
# البحث عن النصوص الأقرب
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
# التطبيق الرئيسي
# =========================================

if uploaded_file is not None:

    with st.spinner("جاري معالجة ملف PDF ..."):

        pdf_text = extract_text_from_pdf(
            uploaded_file
        )

        chunks = split_text(pdf_text)

        index = create_faiss_index(chunks)

    st.success("تمت معالجة الملف بنجاح ✅")

    st.write(f"عدد أجزاء النص: {len(chunks)}")

    # =====================================
    # إدخال السؤال
    # =====================================

    question = st.text_input(
        "اكتب سؤالك هنا:"
    )

    if question:

        with st.spinner("جاري البحث عن الإجابة ..."):

            # البحث في FAISS
            retrieved_chunks = search(
                question,
                index,
                chunks
            )

            # دمج النصوص المسترجعة
            context = "\n\n".join(
                retrieved_chunks
            )

            # بناء Prompt
            prompt = f"""
            أجب فقط باستخدام المعلومات الموجودة في النص التالي.

            إذا لم تجد الإجابة داخل النص قل:
            "المعلومة غير موجودة في الملف"

            النص:
            {context}

            السؤال:
            {question}
            """

            # إرسال الطلب إلى Groq
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "أنت مساعد ذكي للإجابة عن الأسئلة من ملفات PDF."
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
        # عرض الإجابة
        # =====================================

        st.subheader("📌 الإجابة")

        st.write(answer)

        # =====================================
        # عرض النصوص المستخدمة
        # =====================================

        with st.expander("📄 النصوص المستخدمة"):

            for i, chunk in enumerate(retrieved_chunks):

                st.markdown(f"### Chunk {i+1}")

                st.info(chunk)