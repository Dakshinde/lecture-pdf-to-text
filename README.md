# Lecture PDF to Text (Marathi → English → Summary)

A simple **Streamlit web app** that:
1. Extracts handwritten Marathi notes (PDFs or images) using **Google Vision OCR**.
2. Translates them into **English** using **Gemini API**.
3. Summarizes the content into exam-ready notes.

👉 **Live App:** [Click here to open](https://lecture-pdf-to-text-iwfqohxnbkdpz62eafhnqq.streamlit.app/)

---

## 🚀 Features
- 📄 Upload handwritten notes in **PDF, PNG, JPG, JPEG**.
- 🔍 OCR with Google Vision API (supports Marathi handwriting).
- 🌐 Translate Marathi → English.
- ✍️ Summarize into concise, exam-focused notes.
- 💾 Download results as **.txt** or **.docx**.

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – Web interface
- [Google Cloud Vision API](https://cloud.google.com/vision) – OCR
- [Google Gemini API](https://ai.google.dev) – Translation & Summarization
- [PyMuPDF](https://pymupdf.readthedocs.io/) – PDF to image
- [python-docx](https://python-docx.readthedocs.io/) – Save summaries

---

## 🔒 Security
- All **API keys and credentials** are stored securely in **Streamlit Secrets**.
- No sensitive files (like `.env` or service account JSONs) are committed to GitHub.

---

## 👨‍💻 How to Use
1. Open the app from the link above.
2. Upload your handwritten notes (PDF/JPG/PNG).
3. Wait for OCR → Translation → Summary.
4. Download English text or summary.

---

## 📬 Author
Built with ❤️ by **Dakshinde**
