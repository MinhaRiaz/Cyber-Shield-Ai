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
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

NOT_FOUND_RESPONSE = (
    "Sorry, this information is not found in the uploaded legal document."
)

OUT_OF_SCOPE_RESPONSE = (
    "This question is outside the scope of CyberShield AI. "
    "Please ask a question related to cyber safety, cybercrime, "
    "online harassment, digital offences, or the uploaded cyber-law document."
)

CYBER_KEYWORDS = {
    "cyber", "online", "internet", "social media", "facebook", "instagram",
    "whatsapp", "tiktok", "email", "account", "password", "hacking",
    "hack", "phishing", "fraud", "scam", "blackmail", "harassment",
    "harassing", "stalking", "cyberbullying", "bullying", "threat",
    "threatening", "extortion", "impersonation", "privacy", "data",
    "identity", "identity theft", "malware", "virus", "ransomware",
    "doxxing", "fake account", "deepfake", "photo", "video", "defamation",
    "crime", "criminal", "evidence", "report", "complaint", "peca",
    "electronic", "digital", "device", "computer", "mobile", "website",
    "website", "abuse", "blackmailing", "online safety"
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


def is_cyber_question(question: str) -> bool:
    q = question.lower().strip()

    if not q:
        return False

    # Direct legal/cyber terms
    if any(keyword in q for keyword in CYBER_KEYWORDS):
        return True

    # Common question patterns that can be cyber-related
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


def extract_pdf_documents(uploaded_file) -> List[Dict]:
    pdf_bytes = uploaded_file.getvalue()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    documents = []

    for page_number, page in enumerate(pdf, start=1):
        text = clean_text(page.get_text("text"))

        if text:
            documents.append(
                {
                    "source": uploaded_file.name,
                    "page": page_number,
                    "text": text,
                }
            )

    pdf.close()
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


@st.cache_resource(show_spinner="Building search index...")
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
    top_k: int = 5,
    min_similarity: float = 0.30,
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
    technicality: str,
    response_size: str,
) -> str:

    context = format_context(results)

    size_instruction = {
        "Short": "Answer briefly in 3–5 clear bullet points.",
        "Medium": "Give a clear answer with the important legal/safety points.",
        "Detailed": "Give a structured explanation with legal context, practical safety steps, evidence preservation, and reporting guidance where supported.",
    }[response_size]

    technicality_instruction = {
        "Simple": "Use simple language suitable for a general user. Explain legal terms briefly.",
        "Balanced": "Use clear language with necessary legal terminology and short explanations.",
        "Technical": "Use precise legal and technical terminology, while remaining understandable.",
    }[technicality]

    return f"""
You are CyberShield AI, a cyber-safety and Pakistani cyber-law information assistant.

STRICT GROUNDING RULES:
1. Use ONLY the supplied document context for legal claims.
2. Do NOT invent sections, penalties, procedures, authorities, case law, or legal rights.
3. If the supplied context does not support the answer, say exactly:
   "{NOT_FOUND_RESPONSE}"
4. Do not pretend to be a lawyer.
5. Do not provide instructions that facilitate hacking, credential theft, malware,
   evasion, unauthorized access, or other cybercrime.
6. For safety questions, prioritize lawful and defensive actions.
7. If the user appears to face immediate physical danger, advise contacting local
   emergency services or a trusted person rather than focusing only on the law.
8. Clearly distinguish between what the document states and general safety advice.
9. Cite relevant source pages in the answer using [Page X].
10. Never create a citation for a page that is not present in the supplied context.

TECHNICALITY:
{technicality_instruction}

RESPONSE SIZE:
{size_instruction}

USER QUESTION:
{question}

SUPPLIED DOCUMENT CONTEXT:
{context}
"""


def generate_answer(
    question: str,
    results: List[Dict],
    technicality: str,
    response_size: str,
    model_name: str,
    temperature: float,
) -> str:

    if not results:
        return NOT_FOUND_RESPONSE

    api_key = get_groq_api_key()

    if not api_key:
        return (
            "Groq API key is not configured. Add GROQ_API_KEY to Streamlit "
            "Secrets before using the AI response."
        )

    client = Groq(api_key=api_key)

    prompt = build_prompt(
        question,
        results,
        technicality,
        response_size,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful cyber-safety and legal-information assistant. "
                    "Follow the grounding rules exactly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content


# =========================================================
# UI
# =========================================================
st.title("🛡️ CyberShield AI")
st.caption(
    "AI-powered cyber-safety and Pakistani cyber-law information assistant "
    "using Retrieval-Augmented Generation (RAG)."
)

with st.sidebar:
    st.header("⚙️ Settings")

    technicality = st.selectbox(
        "Technicality",
        ["Simple", "Balanced", "Technical"],
        index=1,
    )

    response_size = st.selectbox(
        "Response size",
        ["Short", "Medium", "Detailed"],
        index=1,
    )

    top_k = st.slider(
        "Retrieved sources",
        min_value=2,
        max_value=8,
        value=5,
    )

    min_similarity = st.slider(
        "Minimum similarity",
        min_value=0.10,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help="Higher values make the assistant stricter about document relevance.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=0.6,
        value=0.1,
        step=0.05,
    )

    model_name = st.selectbox(
        "Groq model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        index=0,
    )

    st.divider()

    st.markdown("### 📄 Legal document")
    st.caption(
        "Upload the PECA/cyber-law PDF you want CyberShield AI to use as its "
        "legal knowledge source."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
    )


if not uploaded_file:
    st.info(
        "Upload your PECA / cyber-law PDF from the sidebar to build the RAG index."
    )

    st.markdown("### Example questions")
    st.markdown(
        """
        - What does the law say about cyber stalking?
        - What should I do if someone is harassing me online?
        - What is cyberbullying?
        - What evidence should I preserve?
        - What does the uploaded law say about online threats?
        """
    )

    st.stop()


# =========================================================
# Build RAG corpus
# =========================================================
try:
    documents = extract_pdf_documents(uploaded_file)

    if not documents:
        st.error("No readable text was found in this PDF.")
        st.stop()

    chunked_documents = build_chunks(documents)

    if not chunked_documents:
        st.error("The PDF could not be divided into searchable text chunks.")
        st.stop()

    texts = tuple(item["text"] for item in chunked_documents)
    index = build_faiss_index(texts)

except Exception as e:
    st.error(f"Could not process the PDF: {e}")
    st.stop()


st.success(
    f"Loaded **{len(documents)} pages** and created **{len(chunked_documents)} searchable chunks**."
)


# =========================================================
# Chat
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


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


question = st.chat_input(
    "Describe your cyber-safety or cyber-law question..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if not is_cyber_question(question):
            answer = OUT_OF_SCOPE_RESPONSE
            sources = []

        else:
            sources = retrieve_documents(
                question,
                chunked_documents,
                index,
                top_k=top_k,
                min_similarity=min_similarity,
            )

            answer = generate_answer(
                question,
                sources,
                technicality,
                response_size,
                model_name,
                temperature,
            )

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


with st.expander("ℹ️ How CyberShield AI works"):
    st.markdown(
        """
        **1. Upload PDF** → your selected cyber-law document is read page by page.

        **2. Chunking** → the text is divided into smaller searchable sections.

        **3. Embeddings** → Sentence Transformers converts the sections into vectors.

        **4. FAISS retrieval** → the most relevant sections are retrieved for the question.

        **5. Grounded generation** → Groq generates an answer using the retrieved context.

        **6. Source display** → relevant document pages are shown with the answer.

        The application is designed to reduce unsupported legal claims by refusing to
        answer legal questions when relevant information is not found in the uploaded
        document.
        """
    )

st.caption(
    "⚠️ CyberShield AI provides informational cyber-safety/legal information and is "
    "not a substitute for advice from a qualified lawyer or relevant authority."
)
