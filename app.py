import os
import io
import pymupdf as fitz  # PyMuPDF (modern import)
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="MedReport Simplifier",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Gemini Helpers ───────────────────────────────────────────────────────────

def get_gemini_client(api_key: str) -> genai.Client:
    """Return a configured google-genai Client."""
    return genai.Client(api_key=api_key)


MODEL = "gemini-3.6-flash"


def build_prompt(report_text: str) -> str:
    return f"""You are a compassionate medical communication specialist. A patient has shared their medical report with you. Your task is to convert the clinical language into a clear, warm, and easy-to-understand summary that a person with no medical background can fully understand.

Follow this exact structure in your response:

---

## 🩺 What This Report Is About
Briefly explain what type of medical report this is (e.g., blood test, discharge summary, imaging report) and its general purpose in 1–2 plain-language sentences.

---

## 📋 Key Findings — What Was Found
List the most important findings from the report. For each finding:
- Use everyday language (avoid or define any medical jargon)
- State whether the value/finding is **normal**, **slightly outside normal**, or **requires attention**
- If numbers are involved, include reference ranges in simple terms

---

## ⚠️ Findings That Need Attention
Highlight anything that falls outside normal ranges or that a doctor flagged. Explain *why* it matters in plain language. If everything is normal, say so clearly and reassuringly.

---

## ✅ Recommended Next Steps
Provide clear, actionable steps the patient should consider, such as:
- Follow-up appointments to schedule
- Lifestyle changes (diet, exercise, sleep, hydration)
- Medications or supplements to discuss with their doctor
- Any tests or monitoring that may be needed

---

## 💬 Questions to Ask Your Doctor
Suggest 3–5 specific questions the patient could bring up at their next appointment based on the findings.

---

## 🌟 Reassurance & Summary
End with a brief, supportive paragraph that summarises the overall picture and encourages the patient. Keep the tone warm and empowering.

---

**Important guidelines:**
- Write at a Grade 6–8 reading level.
- Never diagnose or prescribe — remind the patient to consult their healthcare provider for medical decisions.
- If the report is unclear or seems incomplete, note that politely and advise the patient to contact their provider.
- Do NOT use HTML or markdown tables — use bullet points and plain text only.

Here is the medical report to summarise:

---
{report_text}
---"""


def analyse_report(client: genai.Client, report_text: str) -> str:
    """Send report text to Gemini 3.6 Flash and return the simplified summary."""
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(report_text),
    )
    return response.text


def analyse_image_report(
    client: genai.Client, image: Image.Image, extra_context: str = ""
) -> str:
    """Send a PIL image (scanned report page) to Gemini vision and return summary."""
    context_note = (
        f"\n\nAdditional context from the patient: {extra_context}"
        if extra_context
        else ""
    )
    prompt = (
        "You are a compassionate medical communication specialist. "
        "The image below is a page from a medical report. "
        "Extract all text and medical data visible in the image, then produce a "
        "patient-friendly summary following this exact structure:\n\n"
        "## 🩺 What This Report Is About\n"
        "## 📋 Key Findings — What Was Found\n"
        "## ⚠️ Findings That Need Attention\n"
        "## ✅ Recommended Next Steps\n"
        "## 💬 Questions to Ask Your Doctor\n"
        "## 🌟 Reassurance & Summary\n\n"
        "Write at a Grade 6–8 reading level. Never diagnose or prescribe. "
        "Use bullet points, no tables, no HTML."
        + context_note
    )

    # Convert PIL image to bytes for the genai SDK
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        ],
    )
    return response.text


# ─── File Extraction Helpers ──────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, list[Image.Image]]:
    """Extract text and rendered page images from a PDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text: list[str] = []
    images: list[Image.Image] = []

    for page in doc:
        text = page.get_text("text").strip()
        if text:
            full_text.append(text)
        # Render at 2× zoom for legibility in vision calls
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return "\n\n".join(full_text), images


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode a plain-text file trying common encodings."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


# ─── Session State Defaults ───────────────────────────────────────────────────

def init_session_state() -> None:
    defaults = {
        "summary": None,
        "extracted_text": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """Render the sidebar UI and return the active Gemini API key."""
    with st.sidebar:
        st.title("🩺 MedReport Simplifier")
        st.caption("Powered by Gemini 3.6 Flash")
        st.divider()

        st.subheader("🔑 API Configuration")
        env_key = os.getenv("GEMINI_API_KEY", "")

        if env_key:
            st.success("API key loaded from environment (.env file).", icon="✅")
            api_key = env_key
        else:
            api_key = st.text_input(
                "Enter your Gemini API Key",
                type="password",
                placeholder="AIza...",
                help="Get your free key at https://aistudio.google.com/app/apikey",
            )
            if api_key:
                st.success("API key entered.", icon="✅")
            else:
                st.info(
                    "Enter your Gemini API key above, or add a `.env` file "
                    "containing `GEMINI_API_KEY=your_key`.",
                    icon="ℹ️",
                )

        st.divider()
        st.subheader("📂 Supported File Types")
        st.markdown(
            "- **PDF** — Lab reports, discharge summaries, imaging reports\n"
            "- **Images** — PNG, JPG, JPEG (scanned documents)\n"
            "- **Text** — TXT plain-text reports"
        )
        st.divider()
        st.subheader("ℹ️ How It Works")
        st.markdown(
            "1. Enter your Gemini API key\n"
            "2. Upload your medical report\n"
            "3. Optionally add personal context\n"
            "4. Click **Analyse Report**\n"
            "5. Read or download your plain-language summary"
        )
        st.divider()
        st.caption(
            "⚕️ **Disclaimer:** This tool provides simplified summaries for "
            "informational purposes only. It is **not** a substitute for "
            "professional medical advice, diagnosis, or treatment. Always "
            "consult your healthcare provider."
        )

    return api_key


# ─── Main App ─────────────────────────────────────────────────────────────────

def main() -> None:
    init_session_state()
    api_key = render_sidebar()

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🩺 Medical Report Simplifier")
    st.markdown(
        "Upload your medical report and receive a **clear, patient-friendly summary** "
        "written in plain language — no medical jargon, just the information you need."
    )
    st.divider()

    # ── Upload Section ───────────────────────────────────────────────────────
    st.subheader("📤 Upload Your Medical Report")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        help="Accepted formats: PDF, PNG, JPG, JPEG, TXT",
        label_visibility="collapsed",
    )

    extra_context = st.text_area(
        "Optional: Add context about yourself (age, conditions, medications, reason for test)",
        placeholder="Example: I am a 45-year-old woman with Type 2 diabetes. My doctor ordered this test to check my kidney function.",
        height=80,
    )

    if uploaded_file:
        col1, col2, col3 = st.columns(3)
        col1.metric("File", uploaded_file.name)
        col2.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
        col3.metric("Type", uploaded_file.type.split("/")[-1].upper())

    st.divider()

    # ── Analyse Button ───────────────────────────────────────────────────────
    analyse_clicked = st.button(
        "🔍 Analyse Report",
        type="primary",
        disabled=not (uploaded_file and api_key),
        use_container_width=True,
    )

    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar to continue.", icon="⚠️")
    elif not uploaded_file:
        st.info("Upload a medical report above to get started.", icon="📂")

    # ── Processing ───────────────────────────────────────────────────────────
    if analyse_clicked and uploaded_file and api_key:
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.type

        with st.spinner("🔬 Analysing your report with Gemini 3.6 Flash…"):
            try:
                client = get_gemini_client(api_key)
                summary: str | None = None

                # PDF ─────────────────────────────────────────────────────
                if file_type == "application/pdf":
                    text, page_images = extract_text_from_pdf(file_bytes)
                    st.session_state["extracted_text"] = text

                    if text.strip():
                        combined = (
                            f"Patient context: {extra_context}\n\n{text}"
                            if extra_context
                            else text
                        )
                        summary = analyse_report(client, combined)
                    elif page_images:
                        st.info(
                            "No selectable text found — analysing as a scanned document…",
                            icon="🔎",
                        )
                        parts = [
                            analyse_image_report(client, img, extra_context)
                            for img in page_images[:3]
                        ]
                        summary = "\n\n---\n\n".join(parts)
                    else:
                        st.error("Could not extract any content from the PDF.", icon="❌")

                # Images ──────────────────────────────────────────────────
                elif file_type in ("image/png", "image/jpeg", "image/jpg"):
                    image = Image.open(io.BytesIO(file_bytes))
                    st.session_state["extracted_text"] = "[Image-based report — vision analysis]"
                    summary = analyse_image_report(client, image, extra_context)

                # Plain Text ──────────────────────────────────────────────
                elif file_type == "text/plain":
                    text = extract_text_from_txt(file_bytes)
                    st.session_state["extracted_text"] = text
                    combined = (
                        f"Patient context: {extra_context}\n\n{text}"
                        if extra_context
                        else text
                    )
                    summary = analyse_report(client, combined)

                else:
                    st.error(f"Unsupported file type: {file_type}", icon="❌")

                if summary:
                    st.session_state["summary"] = summary

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="❌")
                st.session_state["summary"] = None

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state["summary"]:
        st.divider()
        st.subheader("📝 Your Simplified Report Summary")
        st.markdown(st.session_state["summary"])
        st.divider()

        # Download buttons ────────────────────────────────────────────────
        st.subheader("💾 Download Your Summary")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.download_button(
                label="⬇️ Download as Text (.txt)",
                data=st.session_state["summary"].encode("utf-8"),
                file_name="medical_report_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_dl2:
            md_header = (
                "# Medical Report Summary\n"
                "_Generated by MedReport Simplifier — Gemini 3.6 Flash_\n\n"
            )
            st.download_button(
                label="⬇️ Download as Markdown (.md)",
                data=(md_header + st.session_state["summary"]).encode("utf-8"),
                file_name="medical_report_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Raw text expander ───────────────────────────────────────────────
        raw = st.session_state.get("extracted_text", "")
        if raw and raw != "[Image-based report — vision analysis]":
            with st.expander("🔍 View Extracted Raw Text (from your uploaded file)"):
                st.text_area(
                    "raw_text",
                    value=raw,
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                )

        st.divider()
        if st.button("🔄 Analyse Another Report", use_container_width=True):
            st.session_state["summary"] = None
            st.session_state["extracted_text"] = None
            st.rerun()

    # ── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "MedReport Simplifier · Powered by Google Gemini 3.6 Flash · "
        "For informational purposes only · Not a substitute for professional medical advice"
    )


if __name__ == "__main__":
    main()
