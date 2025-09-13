import os
import io
from typing import List, Tuple

import streamlit as st
import fitz # PyMuPDF
from google.cloud import vision
from google import genai
from google.genai import types as genai_types
from docx import Document
from google.oauth2 import service_account
from streamlit_paste_button import paste_image_button as pbutton
from PIL import Image

# Read secrets from Streamlit Cloud
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEN_MODEL = st.secrets.get("GEN_MODEL", "gemini-2.0-flash")

# Init Gemini client using API key
client = genai.Client(api_key=GEMINI_API_KEY)

# Setup Google Vision API with service account from secrets
GCP_CREDS = st.secrets["gcp_service_account"]
credentials = service_account.Credentials.from_service_account_info(GCP_CREDS)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# Helpers
def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 220) -> List[bytes]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(pix.tobytes("png"))
    return images

def ocr_image_bytes(image_bytes: bytes) -> str:
    image = vision.Image(content=image_bytes)
    ctx = vision.ImageContext(language_hints=["mr", "hi", "en"])
    response = vision_client.document_text_detection(image=image, image_context=ctx)
    if response.error and response.error.message:
        raise RuntimeError(f"Vision error: {response.error.message}")
    return response.full_text_annotation.text or ""

def to_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()

# --- Gemini helpers
def gemini_generate(system_prompt: str, user_prompt: str,
                    max_output_tokens: int = 1500, temperature: float = 0.2) -> str:
    combined = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    resp = client.models.generate_content(
        model=GEN_MODEL,
        contents=combined,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        ),
    )
    return getattr(resp, "text", "")

def translate_marathi_to_english(text: str) -> str:
    system = "You are a precise translator. Translate Marathi to idiomatic English, preserving technical terms."
    user = f"Translate the following Marathi text to English. Reply only with the translated English text.\n\n{text}"
    return gemini_generate(system, user, max_output_tokens=2500, temperature=0.0)

def summarize_english(english_text: str, length_pages: Tuple[int, int] = (1,2)) -> str:
    low, high = length_pages
    system = "You are a study coach. Produce a concise, high-quality summary for exam revision."
    user = (
        f"Summarize the following notes into roughly {low}-{high} pages. Use short paragraphs for topics, "
        "follow each with bullet points of key facts, and end with 3-5 exam-style takeaways.\n\n"
        f"{english_text}"
    )
    return gemini_generate(system, user, max_output_tokens=3000, temperature=0.25)

def answer_question_with_context(question: str, context: str = "") -> str:
    system = "You are a helpful tutor. Answer concisely and show steps where appropriate. If missing info, say so."
    user = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return gemini_generate(system, user, max_output_tokens=800, temperature=0.2)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Lecture Notes Assistant",
    page_icon="📝",
    layout="wide"
)

st.title("📖 Smart Notes Assistant (OCR → Translate → Summarize)")
st.caption("Upload or paste notes, and let AI clean, translate, and summarize them for you.")

# Adding back a simple style for better readability
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class^="st-"], [class*=" st-"] {
        font-family: 'Roboto', sans-serif;
        color: #333;
    }
    .stApp { background-color: #f4f6f8; }
    h1, h2, h3 { color: #1a5276; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True
)

with st.expander("Environment & quick checks"):
    st.write("Using Gemini API key:", bool(GEMINI_API_KEY))
    st.write("Vision creds (GOOGLE_APPLICATION_CREDENTIALS):", bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")))
    st.write("Gemini model:", GEN_MODEL)

# --- DEBUGGING SECTION ---
with st.expander("Paste Button Debug"):
    st.info("After pressing Ctrl+V, check below to see if data is being received.")
    st.write("`paste_result` value:", st.session_state.get('paste_result', 'No paste event yet'))
# --- END DEBUGGING SECTION ---

uploaded = st.file_uploader("Upload handwritten notes (.png/.jpeg/.pdf)", type=["png","jpg","jpeg","pdf"])
dpi = st.slider("OCR DPI", 150, 400, 220)

paste_result = pbutton(label="📋 Paste an image (Ctrl+V after screenshot)")

pages = []
if uploaded:
    raw = uploaded.read()
    if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
        pages = convert_pdf_to_images(raw, dpi=dpi)
    else:
        pages = [raw]
# Use the result from the paste button
elif paste_result.image_data is not None:
    img = paste_result.image_data
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pages = [buf.getvalue()]
    # Store the result for debugging
    st.session_state['paste_result'] = 'Image data received'
    # Force a rerun to process the new image
    st.rerun()
else:
    st.session_state['paste_result'] = 'No image data'


ocr_texts = []
for i, p in enumerate(pages, 1):
    with st.spinner(f"OCR page {i}/{len(pages)}..."):
        try:
            t = ocr_image_bytes(p)
        except Exception as e:
            st.error(f"OCR error on page {i}: {e}")
            t = ""
        ocr_texts.append(t)

combined_text = "\n\n".join(t.strip() for t in ocr_texts if t.strip())
st.subheader("1) Extracted text — edit as needed")
notes_edit = st.text_area("Extracted text (editable)", value=combined_text, height=350)
st.session_state["notes_edit"] = notes_edit

if st.button("➡️ Translate & Summarize"):
    if not notes_edit.strip():
        st.warning("No text to translate.")
    else:
        with st.spinner("Translating (Gemini)..."):
            translation = translate_marathi_to_english(notes_edit)
            st.session_state["translation"] = translation
        with st.spinner("Summarizing (Gemini)..."):
            summary = summarize_english(translation)
            st.session_state["summary"] = summary
            st.success("Done.")

# QA area
st.subheader("Quick QA")
question = st.text_input("Ask a question about these notes")
if st.button("Ask"):
    ctx = st.session_state.get("summary", "") or st.session_state.get("translation", "")
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Answering..."):
            ans = answer_question_with_context(question, ctx)
            st.session_state["last_answer"] = ans
            st.markdown("**💡 Answer:**")
            st.success(ans)

# Outputs and downloads
if st.session_state.get("translation"):
    st.markdown("---")
    st.markdown("### ✨ English Translation")
    st.success(st.session_state["translation"])
    st.download_button("Download translation (.txt)", st.session_state["translation"].encode("utf-8"), file_name="translation.txt")
    st.download_button("Download translation (.docx)", to_docx_bytes(st.session_state["translation"]), file_name="translation.docx")

if st.session_state.get("summary"):
    st.markdown("---")
    st.markdown("### 📌 Summary")
    st.info(st.session_state["summary"])
    st.download_button("Download summary (.txt)", st.session_state["summary"].encode("utf-8"), file_name="summary.txt")
    st.download_button("Download summary (.docx)", to_docx_bytes(st.session_state["summary"]), file_name="summary.docx")

st.caption("Tip: best OCR results at ~300 DPI, dark ink on light background.")
