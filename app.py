import os
import re
from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF


# =========================================================
# Configuration
# =========================================================
st.set_page_config(
    page_title="CyberShield & Legal AI",
    page_icon="🛡️",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# List of embedded legal documents to load into the vector index
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

---
### 🚨 Immediate Reporting & Emergency Guidance
* **National Cyber Crime Investigation Agency (NCCIA)**:
  * **Helpline**: Call **1799** (24/7)
  * **Online Portal**: [complaint.nccia.gov.pk](https://complaint.nccia.gov.pk)
  * **Email**: `helpdesk@nccia.gov.pk`
* **Police & Rescue Emergencies**:
  * **Police Emergency**: Call **15**
  * **Ambulance / Rescue**: Call **1122**
* **Evidence Preservation**: Save unedited screenshots, URLs, chat logs, and timestamps before deleting messages.
"""

# Broadened keyword scope covering PECA + PPC criminal offenses
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


# =========================================================
# Helpers
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
        r"\bonline\b",
        r"\binternet\b",
        r"\bdigital\b",
        r"\belectron",
        r"\bsocial\s+media\b",
        r"\bmy\s+account\b",
        r"\bsomeone\s+is\s+.*\bthreat",
        r"\bsomeone\s+.*\bmessage",
        r"\bsomeone\s+.*\bphoto",
        r"\bsomeone\s+.*\bvideo",
        r"\bsomeone\s+.*\bhurt",
        r"\bwhat\s+should\s+i\s+do\b"
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
) -> str:

    context = format_context(results)

    return f"""
You are CyberShield & Legal AI, an assistant for cyber-safety and Pakistani criminal law (PECA 2016 & PPC 1860).

STRICT GROUNDING RULES:
1. Use ONLY the supplied document context for legal claims and section numbers.
2. Do NOT invent sections, penalties, procedures, or authorities.
3. If the context does not support the answer, state: "{NOT_FOUND_RESPONSE}"
4. Do not pretend to be a lawyer.
5. If the user appears to face immediate physical danger, prioritize urgent safety actions (dialing 15 / 1122 / seeking shelter).
6. Cite relevant source pages in the answer using [Source: Document Name - Page X].

USER QUESTION:
{question}

SUPPLIED DOCUMENT CONTEXT:
{context}
"""


def generate_answer(
    question: str,
    results: List[Dict],
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.1,
) -> str:

    if not results:
        return NOT_FOUND_RESPONSE

    api_key = get_groq_api_key()

    if not api_key:
        return "Groq API key is not configured. Add GROQ_API_KEY to Streamlit Secrets."

    client = Groq(api_key=api_key)

    prompt = build_prompt(question, results)

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
# Sidebar
# =========================================================
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown(
        """
        Welcome to **CyberShield & Legal AI**! 

        1. **Select an Example Question**: Click any predefined query to test PECA or PPC statutes.
        2. **Ask Manually**: Type your question into the chat bar at the bottom.
        3. **View Citations**: Responses feature exact page references from embedded **PECA 2016** and **PPC 1860** statutes.
        """
    )


# =========================================================
# Main UI
# =========================================================
st.title("🛡️ CyberShield & Legal AI")
st.caption(
    "Grounded AI Legal & Cyber-Safety Assistant (PECA 2016 & Pakistan Penal Code 1860)."
)

# Load legal documents
try:
    documents = extract_all_documents(INTERNAL_PDF_FILES)
    if not documents:
        st.error("No legal documents found. Ensure PECA_2016.pdf and PPC_1860.pdf are present.")
        st.stop()
    chunked_documents = build_chunks(documents)
    texts = tuple(item["text"] for item in chunked_documents)
    index = build_faiss_index(texts)
except Exception as e:
    st.error(f"Error initializing legal index: {e}")
    st.stop()


# Example Questions
st.markdown("### 💡 Example Questions")

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

st.divider()

# Message State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['source']} — Page {source['page']}** "
                        f"(similarity: {source['similarity']:.3f})"
                    )

# Manual Input
manual_question = st.chat_input("Ask a question about legal laws (PECA/PPC) or emergency safety...")
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

            raw_answer = generate_answer(question, sources)
            answer = raw_answer + FOOTER_INFO

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.markdown(
                        f"- **{source['source']} — Page {source['page']}** "
                        f"(similarity: {source['similarity']:.3f})"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
