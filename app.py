import os
import re
import base64
from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF


# =========================================================
# Logo Vector SVG (Embedded Inline)
# =========================================================
CYBERSHIELD_SVG_LOGO = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>

    <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="50%" stop-color="#0284c7" />
      <stop offset="100%" stop-color="#0369a1" />
    </linearGradient>

    <filter id="cyanGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>

    <pattern id="circuit" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 10 0 L 10 20 L 20 20" fill="none" stroke="#1e3a8a" stroke-width="1.5" opacity="0.4"/>
      <circle cx="20" cy="20" r="2" fill="#1e3a8a" opacity="0.6"/>
    </pattern>
  </defs>

  <rect width="500" height="500" rx="40" fill="url(#bgGrad)"/>
  <rect width="500" height="500" rx="40" fill="url(#circuit)"/>

  <path d="M 250 70 
           C 340 70, 390 90, 410 130 
           C 410 250, 360 360, 250 430 
           C 140 360, 90 250, 90 130 
           C 110 90, 160 70, 250 70 Z" 
        fill="none" 
        stroke="#38bdf8" 
        stroke-width="5" 
        filter="url(#cyanGlow)"
        opacity="0.8"/>

  <path d="M 250 85 
           C 330 85, 375 102, 392 138 
           C 392 242, 347 342, 250 408 
           C 153 342, 108 242, 108 138 
           C 125 102, 170 85, 250 85 Z" 
        fill="url(#shieldGrad)" 
        stroke="#0284c7" 
        stroke-width="3"/>

  <path d="M 250 105 
           C 315 105, 355 120, 370 150 
           C 370 230, 330 315, 250 375 
           C 170 315, 130 230, 130 150 
           C 145 120, 185 105, 250 105 Z" 
        fill="none" 
        stroke="#7dd3fc" 
        stroke-width="2" 
        opacity="0.5"/>

  <path d="M 250 170 L 250 320" stroke="#ffffff" stroke-width="8" stroke-linecap="round"/>
  <path d="M 210 320 L 290 320" stroke="#ffffff" stroke-width="8" stroke-linecap="round"/>
  <path d="M 170 200 L 330 200" stroke="#ffffff" stroke-width="6" stroke-linecap="round"/>
  <path d="M 170 200 L 140 250 L 200 250 Z" fill="none" stroke="#ffffff" stroke-width="4" stroke-linejoin="round"/>
  <path d="M 330 200 L 300 250 L 360 250 Z" fill="none" stroke="#ffffff" stroke-width="4" stroke-linejoin="round"/>

  <circle cx="250" cy="200" r="22" fill="#0f172a" stroke="#ffffff" stroke-width="4"/>
  <circle cx="250" cy="200" r="8" fill="#38bdf8"/>

  <circle cx="170" cy="200" r="5" fill="#38bdf8"/>
  <circle cx="330" cy="200" r="5" fill="#38bdf8"/>
  <line x1="250" y1="140" x2="250" y2="170" stroke="#38bdf8" stroke-width="3" stroke-dasharray="3 3"/>
  <circle cx="250" cy="135" r="4" fill="#38bdf8"/>
</svg>
"""

# Helper to render SVG string as base64 image tag for Streamlit HTML
def render_svg_as_html(svg_string: str, width: int = 60) -> str:
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{width}px" style="vertical-align: middle;" />'


# =========================================================
# Page & Layout Configuration
# =========================================================
st.set_page_config(
    page_title="CyberShield & Legal AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI/UX Styling
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        margin-bottom: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .header-text {
        flex-grow: 1;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
        color: #38bdf8;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 0px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 10px;
        background-color: #0284c7;
        color: white;
    }
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .emergency-card {
        background-color: #0f172a;
        border-left: 5px solid #ef4444;
        padding: 16px;
        border-radius: 8px;
        margin-top: 20px;
        border: 1px solid #1e293b;
    }
    hr {
        margin: 1.5rem 0;
        border-color: #334155;
    }
    </style>
""", unsafe_allow_html=True)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERNAL_PDF_FILES = ["PECA_2016.pdf", "PPC_1860.pdf"]

NOT_FOUND_RESPONSE = (
    "Sorry, this information is not found in the referenced legal documents (PECA / PPC)."
)

OUT_OF_SCOPE_RESPONSE = (
    "This question appears outside the scope of CyberShield & Legal AI. "
    "Please ask a question related to cyber safety, criminal offences under PPC, "
    "online harassment, or Pakistani legal statutes."
)

FOOTER_INFO = """
<div class="emergency-card">
<h4>🚨 Immediate Reporting & Emergency Guidance</h4>
<ul>
  <li><b>National Cyber Crime Investigation Agency (NCCIA)</b>: Helpline <b>1799</b> (24/7) | <a href="https://complaint.nccia.gov.pk" target="_blank">Online Portal</a></li>
  <li><b>Digital Rights Foundation (DRF) Helpline</b>: <b>0800-39393</b> (Free & Confidential Support)</li>
  <li><b>Police & Emergency Rescue</b>: Police <b>15</b> | Rescue <b>1122</b></li>
  <li><b>Evidence Preservation</b>: Save unedited screenshots, URLs, chat logs, and timestamps before deleting any messages.</li>
</ul>
</div>
"""

LEGAL_KEYWORDS = {
    "cyber", "online", "internet", "social media", "facebook", "instagram",
    "whatsapp", "tiktok", "email", "account", "password", "hacking",
    "hack", "phishing", "fraud", "scam", "blackmail", "harassment",
    "harassing", "stalking", "cyberbullying", "bullying", "threat",
    "threatening", "extortion", "impersonation", "privacy", "data",
    "identity", "identity theft", "malware", "virus", "ransomware",
    "doxxing", "fake account", "deepfake", "photo", "video", "defamation",
    "crime", "criminal", "evidence", "report", "complaint", "peca", "ppc",
    "electronic", "digital", "device", "computer", "mobile", "website",
    "abuse", "blackmailing", "online safety", "rape", "assault", "murder",
    "hurt", "kidnap", "abduction", "theft", "dacoity", "police", "section",
    "punishment", "imprisonment", "fine", "law", "legal", "offence"
}

SUPPORTED_CATEGORIES = {
    "📱 Digital Harassment & Cyberstalking": [
        "Cyber stalking (repeating unwanted contact online) — PECA Sec 24",
        "Cyberbullying & online harassment — PECA Sec 24A",
        "Doxxing & releasing private personal details/photos without consent",
        "Online threats & blackmailing via chat/WhatsApp/social media"
    ],
    "🔐 Account Security & Unauthorized Access": [
        "Account hacking & unauthorized access to messages/data — PECA Sec 3",
        "Identity theft & unauthorized use of identity information — PECA Sec 16",
        "Fake social media accounts & impersonation",
        "Unauthorized SIM card issuance — PECA Sec 17"
    ],
    "💸 Financial Scams & Digital Fraud": [
        "Online bank fraud & OTP/password scams — PECA Sec 14",
        "Phishing websites, spoofing & counterfeit platforms — PECA Sec 26",
        "Spamming & fraudulent marketing messages — PECA Sec 25",
        "Extortion & unauthorized data copying — PECA Sec 4"
    ],
    "🚨 Offenses Against Person (PPC Criminal Laws)": [
        "Physical threats, intimidation & assault — PPC Sections",
        "Rape, sexual assault & offences against modesty — PPC Sec 375/376 & PECA Sec 21",
        "Kidnapping, abduction & human trafficking — PPC Sections & PECA Sec 22C",
        "Child protection against online exploitation — PECA Sec 22/22A"
    ]
}


# =========================================================
# Helpers & RAG Processing
# =========================================================
def get_groq_api_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("GROQ_API_KEY", "")

    return key


def is_legal_or_cyber_question(question: str) -> bool:
    q = question.lower().strip()

    if not q:
        return False

    if any(keyword in q for keyword in LEGAL_KEYWORDS):
        return True

    patterns = [
        r"\bonline\b", r"\binternet\b", r"\bdigital\b", r"\belectron",
        r"\bsocial\s+media\b", r"\bmy\s+account\b", r"\bsomeone\s+is\s+.*\bthreat",
        r"\bsomeone\s+.*\bmessage", r"\bsomeone\s+.*\bphoto", r"\bsomeone\s+.*\bvideo",
        r"\bsomeone\s+.*\bhurt", r"\bwhat\s+should\s+i\s+do\b"
    ]

    return any(re.search(pattern, q) for pattern in patterns)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def extract_all_documents(pdf_filenames: List[str]) -> List[Dict]:
    documents = []
    missing_files = []

    for filename in pdf_filenames:
        file_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
            continue

        pdf = fitz.open(file_path)
        for page_number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                documents.append(
                    {
                        "source": filename,
                        "page": page_number,
                        "text": text,
                    }
                )
        pdf.close()

    if missing_files:
        st.warning(f"Missing legal PDF files in project directory: {', '.join(missing_files)}")

    return documents


def build_chunks(documents: List[Dict]) -> List[Dict]:
    chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunks.append(
                {
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": chunk,
                }
            )
    return chunks


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Indexing legal databases (PECA & PPC)...")
def build_faiss_index(texts: Tuple[str, ...]):
    model = load_embedding_model()

    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_documents(
    question: str,
    chunked_documents: List[Dict],
    index,
    top_k: int = 6,
    min_similarity: float = 0.25,
) -> List[Dict]:

    if not chunked_documents or index is None:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunked_documents)),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        similarity = float(score)

        if similarity >= min_similarity:
            item = dict(chunked_documents[idx])
            item["similarity"] = similarity
            results.append(item)

    return results


def format_context(results: List[Dict]) -> str:
    blocks = []
    for i, item in enumerate(results, start=1):
        blocks.append(
            f"[Source {i}]\n"
            f"Document: {item['source']}\n"
            f"Page: {item['page']}\n"
            f"Similarity: {item['similarity']:.3f}\n"
            f"Text: {item['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    results: List[Dict],
    response_format: str = "Full Explanation",
) -> str:

    context = format_context(results)

    if response_format == "Main Points (Bullets)":
        format_instruction = (
            "Provide the response strictly as clear, bulleted main points highlighting key legal takeaways, "
            "penalties, and immediate actionable steps."
        )
    elif response_format == "Brief Summary":
        format_instruction = (
            "Provide a concise 3 to 4 sentence summary explaining the situation, legal standing, and key action steps."
        )
    else:  # Full Explanation
        format_instruction = (
            "Provide a detailed, comprehensive legal and practical explanation broken down into structured sections."
        )

    return f"""
You are CyberShield & Legal AI, an assistant for cyber-safety and Pakistani criminal law (PECA 2016 & PPC 1860).

STRICT GROUNDING RULES:
1. Use ONLY the supplied document context for legal claims and section numbers.
2. Do NOT invent sections, penalties, procedures, or authorities.
3. If the context does not support the answer, state: "{NOT_FOUND_RESPONSE}"
4. Do not pretend to be a lawyer.
5. If the user appears to face immediate physical danger, prioritize urgent safety actions (dialing 15 / 1122 / seeking shelter).
6. Cite relevant source pages in the answer using [Source: Document Name - Page X].

RESPONSE FORMAT INSTRUCTION:
{format_instruction}

USER QUESTION:
{question}

SUPPLIED DOCUMENT CONTEXT:
{context}
"""


def generate_answer(
    question: str,
    results: List[Dict],
    response_format: str = "Full Explanation",
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.1,
) -> str:

    if not results:
        return NOT_FOUND_RESPONSE

    api_key = get_groq_api_key()

    if not api_key:
        return "Groq API key is not configured. Add GROQ_API_KEY to Streamlit Secrets."

    client = Groq(api_key=api_key)
    prompt = build_prompt(question, results, response_format)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a legal assistant for PECA and PPC statutes."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content


# =========================================================
# Sidebar UI (Featuring Inline SVG Logo)
# =========================================================
with st.sidebar:
    # Sidebar Header with SVG Logo
    st.markdown(
        f"""
        <div class="sidebar-header">
            {render_svg_as_html(CYBERSHIELD_SVG_LOGO, width=48)}
            <div>
                <h2 style="margin: 0; font-size: 1.4rem; color: #38bdf8;">CyberShield AI</h2>
                <span style="font-size: 0.8rem; color: #94a3b8;">v2.0 • RAG Assistant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.subheader("⚙️ Response Style")
    response_format = st.radio(
        "Choose Answer Depth:",
        ["Full Explanation", "Main Points (Bullets)", "Brief Summary"],
        index=0,
        help="Controls the level of detail provided by the AI assistant."
    )
    
    st.markdown("---")
    st.subheader("📋 Quick Scope Checker")
    selected_cat = st.selectbox("Select a Topic to Check Scope:", list(SUPPORTED_CATEGORIES.keys()))
    
    st.markdown(f"**Issues covered under {selected_cat}:**")
    for issue in SUPPORTED_CATEGORIES[selected_cat]:
        st.markdown(f"• {issue}")


# =========================================================
# Main UI & Header Banner (Featuring Inline SVG Logo)
# =========================================================
st.markdown(
    f"""
    <div class="header-card">
        <div>
            {render_svg_as_html(CYBERSHIELD_SVG_LOGO, width=85)}
        </div>
        <div class="header-text">
            <div class="header-title">CyberShield & Legal AI</div>
            <div class="header-subtitle">Grounded Cyber-Safety, Cyberbullying & Criminal Law Assistant for Pakistan</div>
            <div>
                <span class="badge">PECA 2016 Indexed</span>
                <span class="badge">PPC 1860 Indexed</span>
                <span class="badge">Groq LPUs Active</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Load legal documents
try:
    documents = extract_all_documents(INTERNAL_PDF_FILES)
    if not documents:
        st.error("No legal documents found. Ensure PECA_2016.pdf and PPC_1860.pdf are present in the directory.")
        st.stop()
    chunked_documents = build_chunks(documents)
    texts = tuple(item["text"] for item in chunked_documents)
    index = build_faiss_index(texts)
except Exception as e:
    st.error(f"Error initializing legal vector index: {e}")
    st.stop()


# Expandable Supported Issues Overview Section
with st.expander("📌 **Click here to view ALL Supported Issues & Covered Laws (PECA & PPC)**", expanded=False):
    st.markdown("### What issues can you ask CyberShield AI about?")
    st.caption("The system can answer legal questions, penalties, and defensive steps for all the following categories:")
    
    cols = st.columns(2)
    cat_keys = list(SUPPORTED_CATEGORIES.keys())
    
    for idx, key in enumerate(cat_keys):
        col = cols[idx % 2]
        with col:
            st.markdown(f"#### {key}")
            for item in SUPPORTED_CATEGORIES[key]:
                st.markdown(f"- {item}")
            st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# Example Questions Section
st.subheader("💡 Frequently Asked Example Questions")

example_questions = [
    "What does Section 24A of PECA say about cyberbullying?",
    "What are the penalties under PPC for rape and sexual assault?",
    "What is the punishment for cyber stalking under PECA?",
    "What does the Pakistan Penal Code say about criminal intimidation?",
    "How can I report online harassment or physical threats?",
]

col1, col2 = st.columns(2)
selected_example = None

for i, eq in enumerate(example_questions):
    col = col1 if i % 2 == 0 else col2
    if col.button(eq, use_container_width=True):
        selected_example = eq

st.markdown("---")

# Message State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

        if message.get("sources"):
            with st.expander("📚 Verified Statutory Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['source']} — Page {source['page']}** "
                        f"*(Relevance Score: {source['similarity']:.3f})*"
                    )

# Chat Input Handler
manual_question = st.chat_input("Ask a question about legal laws (PECA/PPC), cyber safety, or reporting...")
question = selected_example or manual_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not is_legal_or_cyber_question(question):
            answer = OUT_OF_SCOPE_RESPONSE + FOOTER_INFO
            sources = []
        else:
            sources = retrieve_documents(
                question,
                chunked_documents,
                index,
                top_k=6,
                min_similarity=0.25,
            )

            raw_answer = generate_answer(
                question,
                sources,
                response_format=response_format
            )
            answer = raw_answer + FOOTER_INFO

        st.markdown(answer, unsafe_allow_html=True)

        if sources:
            with st.expander("📚 Verified Statutory Sources"):
                for source in sources:
                    st.markdown(
                        f"- **{source['source']} — Page {source['page']}** "
                        f"*(Relevance Score: {source['similarity']:.3f})*"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
