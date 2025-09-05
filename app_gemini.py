# app_gemini.py
"""
Streamlit app: Upload Marathi notes (image/pdf) -> Vision OCR -> Edit Marathi ->
Translate+Summarize+QA with Gemini Developer API (api key).
Run: streamlit run app_gemini.py
"""

import os
import io
from typing import List, Tuple
import base64

import streamlit as st
import fitz  # PyMuPDF
from google.cloud import vision
from google import genai
from google.genai import types as genai_types
from docx import Document
from dotenv import load_dotenv

# load .env in dev
load_dotenv()

# Config / credentials (Gemini API key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.0-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Get one from Google AI Studio and set GEMINI_API_KEY in your env or .env file. See README.")

# Init Gemini client using Developer API key (consumer)
client = genai.Client(api_key=GEMINI_API_KEY)

# Vision client uses GOOGLE_APPLICATION_CREDENTIALS (service account)
vision_client = vision.ImageAnnotatorClient()

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

# --- Gemini helpers (simple wrapper)
def gemini_generate(system_prompt: str, user_prompt: str,
                    max_output_tokens: int = 1500, temperature: float = 0.2) -> str:
    """Call Google GenAI simply by sending a combined prompt string."""
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

def translate_marathi_to_english(marathi_text: str) -> str:
    system = "You are a precise translator. Translate Marathi to idiomatic English, preserving technical terms."
    user = f"Translate the following Marathi text to English. Reply only with the translated English text.\n\n{marathi_text}"
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
st.set_page_config(page_title="Marathi Notes → Gemini (API key)", layout="wide")
st.title("Marathi Lecture Notes → Gemini (translate & summarize)")

with st.expander("Environment & quick checks"):
    st.write("Using Gemini API key:", bool(GEMINI_API_KEY))
    st.write("Vision creds (GOOGLE_APPLICATION_CREDENTIALS):", bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")))
    st.write("Gemini model:", GEN_MODEL)

uploaded = st.file_uploader("Upload handwritten notes (.png/.jpeg/.pdf)", type=["png","jpg","jpeg","pdf"])
dpi = st.slider("OCR DPI", 150, 400, 220)

if uploaded:
    raw = uploaded.read()
    pages = []
    if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
        pages = convert_pdf_to_images(raw, dpi=dpi)
    else:
        pages = [raw]

    ocr_texts = []
    for i, p in enumerate(pages, 1):
        with st.spinner(f"OCR page {i}/{len(pages)}..."):
            try:
                t = ocr_image_bytes(p)
            except Exception as e:
                st.error(f"OCR error on page {i}: {e}")
                t = ""
            ocr_texts.append(t)

    combined_marathi = "\n\n".join(t.strip() for t in ocr_texts if t.strip())
    st.subheader("1) Extracted Marathi text — edit as needed")
    marathi_edit = st.text_area("Marathi (editable)", value=combined_marathi, height=350)
    st.session_state["marathi_edit"] = marathi_edit

    if st.button("➡️ Translate & Summarize"):
        if not marathi_edit.strip():
            st.warning("No text to translate.")
        else:
            with st.spinner("Translating (Gemini)..."):
                translation = translate_marathi_to_english(marathi_edit)
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
                st.markdown("**Answer:**")
                st.write(ans)

# Outputs and downloads
if st.session_state.get("translation"):
    st.markdown("---")
    st.subheader("English Translation")
    st.write(st.session_state["translation"])
    st.download_button("Download translation (.txt)", st.session_state["translation"].encode("utf-8"), file_name="translation.txt")
    st.download_button("Download translation (.docx)", to_docx_bytes(st.session_state["translation"]), file_name="translation.docx")

if st.session_state.get("summary"):
    st.markdown("---")
    st.subheader("Summary")
    st.write(st.session_state["summary"])
    st.download_button("Download summary (.txt)", st.session_state["summary"].encode("utf-8"), file_name="summary.txt")
    st.download_button("Download summary (.docx)", to_docx_bytes(st.session_state["summary"]), file_name="summary.docx")

st.caption("Tip: best OCR results at ~300 DPI, dark ink on light background.")
