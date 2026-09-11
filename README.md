# 🛡️ CyberShield AI

CyberShield AI is an AI-powered **cyber-safety and Pakistani cyber-law information assistant** built with Retrieval-Augmented Generation (RAG).

The application internally processes the **Prevention of Electronic Crimes Act (PECA 2016)** document, allowing users to select example questions or manually ask cyber-safety questions directly without uploading a file.

## Features

- 📄 Embedded PECA 2016 cyber-law reference document
- 💡 Example questions for instant querying
- 💬 Chat interface for manual question entry
- 🔎 Semantic retrieval with Sentence Transformers
- ⚡ FAISS vector search
- 🤖 Groq-powered answer generation
- 🛡️ Cyber-question scope detection
- 📚 Automatic source and page citations

## Project structure

```text
CyberShield-AI/
├── app.py
├── requirements.txt
├── PECA_2016.pdf
└── README.md
