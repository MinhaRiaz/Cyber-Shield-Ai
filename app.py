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
LOCAL_PDF_PATH = "PECA_2016.pdf"

NOT_FOUND_RESPONSE = (
    "Sorry, this information is not found in the uploaded legal document."
)

OUT_OF_SCOPE_RESPONSE = (
    "This question is outside the scope of CyberShield AI. "
    "Please ask a question related to cyber safety, cybercrime, "
    "online harassment, digital offences, or the PECA cyber-law document."
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
    "abuse", "blackmailing", "online safety"
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

    if any(keyword in q for keyword in CYBER_KEYWORDS):
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


def extract_pdf_documents_from_filepath(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        st.error(f"Embedded PDF file not found at path: `{file_path}`. Please ensure `{file_path}` exists in the working directory.")
        st.stop()

    pdf = fitz.open(file_path)
    documents = []

    for page_number, page in enumerate(pdf, start=1):
        text = clean_text(page.get_text("text"))

        if text:
            documents.append(
                {
                    "source": os.path.basename(file_path),
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
) -> str:

    context = format_context(results)

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
        return (
            "Groq API key is not configured. Add GROQ_API_KEY to Streamlit "
            "Secrets or set environment variable before requesting answers."
        )

    client = Groq(api_key=api_key)

    prompt = build_prompt(
        question,
        results,
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
# Sidebar: How To Use
# =========================================================
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown(
        """
        Welcome to **CyberShield AI**! Follow these steps to ask legal or cyber-safety questions:

        1. **Select an Example Question**: Click on any of the example question buttons to ask predefined legal queries.
        2. **Ask Manually**: Type your own cyber-safety or cyber-law query in the input box at the bottom.
        3. **View Source References**: Each response includes verifiable page citations from the embedded **Prevention of Electronic Crimes Act (PECA 2016)** document.
        
        ---
        🔒 *Note: The system searches directly within the internal PECA 2016 law database.*
        """
    )


# =========================================================
# Main UI
# =========================================================
st.title("🛡️ CyberShield AI")
st.caption(
    "AI-powered cyber-safety and Pakistani cyber-law information assistant built with RAG."
)

# Load embedded PDF documents automatically
try:
    documents = extract_pdf_documents_from_filepath(LOCAL_PDF_PATH)
    chunked_documents = build_chunks(documents)
    texts = tuple(item["text"] for item in chunked_documents)
    index = build_faiss_index(texts)
except Exception as e:
    st.error(f"Error loading internal legal document: {e}")
    st.stop()


# Example Questions Section
st.markdown("### 💡 Example Questions")
st.caption("Click a button below to quickly run a sample query:")

example_questions = [
    "What does the law say about cyber stalking?",
    "What is the punishment for cyberbullying?",
    "What does the law say about online threats?",
    "What is electronic fraud according to PECA?",
    "What evidence should I preserve in case of cybercrime?",
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

# Manual Question Input
manual_question = st.chat_input("Ask a question about cyber-law or online safety...")

# Handle input priority (either example button clicked or manually submitted)
question = selected_example or manual_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})

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
                top_k=5,
                min_similarity=0.30,
            )

            answer = generate_answer(
                question,
                sources,
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
