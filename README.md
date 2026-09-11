# 🛡️ CyberShield AI

CyberShield AI is an AI-powered **cyber-safety and Pakistani cyber-law information assistant** built with Retrieval-Augmented Generation (RAG).

The application lets the user upload a cyber-law PDF, ask a question, retrieve the most relevant passages with FAISS, and generate a grounded answer using Groq.

## Features

- 📄 Upload a PECA / cyber-law PDF directly in the app
- 🔎 Semantic retrieval with Sentence Transformers
- ⚡ FAISS vector search
- 🤖 Groq-powered answer generation
- 🛡️ Cyber-question scope detection
- 🚫 Not-found protection
- 📚 Source and page references
- 🎚️ Technicality control
- 📏 Response-size control
- 🔢 Top-K retrieval control
- 🎯 Minimum similarity threshold
- 🌡️ Temperature control
- 🔐 Groq API key through Streamlit Secrets
- ☁️ Ready for Streamlit Community Cloud
- 💻 Can also run locally

## Project structure

```text
CyberShield-AI/
├── app.py
├── requirements.txt
└── README.md
```

Only these three files are required.

The legal PDF does **not** need to be committed to GitHub. The user uploads the PDF through the application interface.

## RAG pipeline

```text
Legal PDF
   ↓
PDF text extraction
   ↓
Cleaning
   ↓
Text chunking
   ↓
Sentence Transformer embeddings
   ↓
FAISS similarity search
   ↓
Relevant legal context
   ↓
Groq LLM
   ↓
Grounded cyber-safety answer
   ↓
Source/page references
```

## Local setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key.

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### Linux / macOS

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Run:

```bash
streamlit run app.py
```

Then upload your PECA/cyber-law PDF.

## Streamlit Community Cloud deployment

1. Create a GitHub repository.
2. Add exactly these files:

```text
app.py
requirements.txt
README.md
```

3. Open Streamlit Community Cloud.
4. Create a new app and select your GitHub repository.
5. Set the main file to:

```text
app.py
```

6. Open the app's **Secrets** settings.
7. Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

8. Deploy the application.
9. Upload your PECA/cyber-law PDF through the sidebar.

## Important security rule

**Never put the Groq API key directly inside `app.py` or upload it to GitHub.**

Use Streamlit Secrets for deployment.

## Legal grounding

CyberShield AI is designed to answer legal-information questions only from the context retrieved from the uploaded document.

If relevant information cannot be retrieved, the application returns:

> Sorry, this information is not found in the uploaded legal document.

The application also avoids generating instructions that could facilitate hacking, unauthorized access, credential theft, malware, or other cybercrime.

## Recommended PECA questions

After uploading the PECA document, try:

- What does the law say about cyber stalking?
- What is cyberbullying?
- What should I do if someone is harassing me online?
- What evidence should I preserve?
- What does the law say about online threats?
- What does the uploaded document say about electronic offences?

## Testing

### Out-of-scope test

Question:

```text
What is the weather today?
```

Expected behavior:

```text
This question is outside the scope of CyberShield AI...
```

### In-scope test

Question:

```text
What is cyber stalking?
```

Expected behavior:

- Retrieve relevant legal passages.
- Generate a grounded answer.
- Show the relevant PDF page/source.

### Not-found test

Ask a cyber-law question whose answer is not supported by the uploaded document.

Expected behavior:

```text
Sorry, this information is not found in the uploaded legal document.
```

## Limitations

- The legal answer is only as reliable as the uploaded source document and retrieval quality.
- The system is an informational assistant, not a lawyer.
- Legal documents can be amended, repealed, or superseded. Always verify important legal matters against the latest authoritative source.
- The first deployment can take longer because the Sentence Transformer model may need to be downloaded and loaded.

## Technology stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Groq
- PyMuPDF
- NumPy

## Disclaimer

CyberShield AI provides general informational assistance and cyber-safety guidance. It does not establish an attorney-client relationship and should not be treated as a substitute for professional legal advice or instructions from the relevant authorities.
