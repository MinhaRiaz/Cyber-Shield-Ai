# 🛡️ CyberShield AI

### Grounded Cyber-Safety, Cyberbullying & Criminal Law Assistant for Pakistan

**CyberShield AI** is an AI-powered Retrieval-Augmented Generation (RAG) application designed to provide accessible, grounded information about **cyber safety, cybercrime, online harassment, and criminal offences under Pakistani law**.

The application uses **PECA 2016** and the **Pakistan Penal Code (PPC) 1860** as its internal legal knowledge base. Relevant sections are retrieved using semantic search and provided to an AI model before generating a response.

> ⚠️ **Disclaimer:** CyberShield & Legal AI is an informational AI assistant. It does not provide professional legal advice and should not replace a qualified lawyer, law-enforcement authority, or official government guidance.

---

## 🌐 Live Demo

🚀 **Try the application:**
https://cyber-shield-ai.streamlit.app/

---

## 🎯 Problem

Cybercrime and online abuse are increasingly common, but many people do not know:

* What type of cyber or criminal offence they may be facing
* Which Pakistani laws may be relevant
* What evidence they should preserve
* What immediate steps they should take
* How to report an incident
* What actions they should avoid

Legal documents can also be difficult for non-lawyers to understand.

CyberShield & Legal AI aims to bridge this gap by combining **semantic document retrieval, Pakistani legal references, and Generative AI** into a simple conversational interface.

---

## 💡 Solution

The user describes their situation or asks a legal question in natural language.

CyberShield AI then:

1. Checks whether the question is within the supported cyber/legal scope.
2. Searches the indexed PECA 2016 and PPC 1860 documents.
3. Retrieves the most relevant legal passages.
4. Sends the retrieved context to the Groq-hosted LLM.
5. Generates a grounded response.
6. Provides document and page references for retrieved sources.
7. Adds practical safety, evidence-preservation, and reporting guidance where appropriate.

The system is designed to reduce unsupported legal claims by instructing the LLM to rely on the retrieved statutory context.

---

# 🧠 Retrieval-Augmented Generation (RAG)

CyberShield AI uses the following RAG pipeline:

```text
                User Question
                     │
                     ▼
          Scope / Legal Query Check
                     │
                     ▼
              Query Embedding
                     │
                     ▼
             FAISS Similarity Search
                     │
                     ▼
       Relevant PECA / PPC Text Chunks
                     │
                     ▼
             Groq LLM Generation
                     │
                     ▼
          Grounded Legal Response
                     │
                     ▼
           Source & Page References
```

### RAG Process

#### 1. Legal Documents

The application loads two internal PDF knowledge sources:

* `PECA_2016.pdf`
* `PPC_1860.pdf`

#### 2. PDF Text Extraction

Text is extracted from each PDF page using **PyMuPDF**.

Each extracted page is stored together with:

* Document name
* Page number
* Extracted text

#### 3. Text Cleaning

Whitespace is normalized before the text is processed.

#### 4. Chunking

Legal text is divided into smaller chunks using a word-based chunking strategy.

Current configuration:

```text
Chunk Size: 900 words
Overlap: 150 words
```

The overlap helps preserve context between neighboring chunks.

#### 5. Embeddings

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert legal text into vector embeddings.

Embeddings are normalized before indexing.

#### 6. FAISS

FAISS is used for vector similarity search.

The application uses:

```text
IndexFlatIP
```

with normalized embeddings, allowing efficient inner-product similarity search.

#### 7. Retrieval

For each user question:

```text
Top K = 6
Minimum Similarity = 0.25
```

The most relevant legal chunks are retrieved.

#### 8. Grounded Generation

The retrieved context is passed to:

```text
Groq
openai/gpt-oss-120b
```

The model is instructed to:

* Use only supplied legal context for legal claims
* Avoid inventing sections or penalties
* Cite source documents and pages
* Avoid pretending to be a lawyer
* Prioritize immediate safety when physical danger is involved

---

# ⚖️ Legal Knowledge Base

CyberShield AI currently indexes:

### PECA 2016

**Prevention of Electronic Crimes Act, 2016**

Used for cyber-related issues including areas such as:

* Unauthorized access
* Identity-related offences
* Online harassment
* Cyberstalking
* Electronic fraud
* Spoofing
* Spamming
* Extortion
* Cyber-related offences against individuals
* Other electronic crimes covered by the indexed document

### PPC 1860

**Pakistan Penal Code, 1860**

Used for criminal-law questions involving areas such as:

* Criminal intimidation
* Assault
* Hurt
* Theft
* Dacoity
* Kidnapping
* Abduction
* Rape
* Offences against modesty
* Other criminal offences covered by the indexed document

> The application uses the uploaded legal documents as its retrieval knowledge base. Users should verify important legal matters against current official sources or qualified legal professionals.

---

# ✨ Features

## 🔎 Natural Language Legal Questions

Users can describe their situation normally without knowing the exact legal terminology.

Example:

> Someone is threatening me on WhatsApp and says they will share my private photos.

The application retrieves relevant legal context and generates a structured response.

---

## 📚 Verified Statutory Sources

Retrieved responses can show:

```text
Document Name
Page Number
Relevance Score
```

Example:

```text
PECA_2016.pdf — Page 24
Relevance Score: 0.731
```

This helps users understand where the retrieved information came from.

---

## 🛡️ Cyber-Safety Guidance

The system supports questions involving:

* Online harassment
* Cyberstalking
* Cyberbullying
* Blackmail
* Threats
* Fake accounts
* Impersonation
* Hacking
* Phishing
* Online fraud
* Identity theft
* Doxxing
* Privacy-related issues
* Digital abuse

---

## 🚨 Criminal-Law Guidance

The application also supports selected criminal-law topics under PPC, including:

* Physical threats
* Intimidation
* Assault
* Hurt
* Rape
* Sexual offences
* Kidnapping
* Abduction
* Theft
* Dacoity
* Other criminal offences contained in the indexed PPC document

---

## 🧾 Evidence Preservation

CyberShield AI provides evidence-preservation guidance, including recommendations to preserve:

* Unedited screenshots
* URLs
* Chat logs
* Usernames
* Dates
* Timestamps
* Relevant digital records

Users are encouraged to preserve potentially important evidence before deleting messages or accounts.

---

## 🚑 Immediate Safety Guidance

For situations involving immediate physical danger, the application prioritizes safety guidance.

The interface provides:

* **Police: 15**
* **Rescue: 1122**
* **NCCIA Helpline: 1799**
* NCCIA online complaint portal
* Digital Rights Foundation helpline information

> Emergency information should always be independently verified because contact details and official procedures can change.

---

# 📋 Supported Categories

CyberShield AI currently organizes supported issues into four major categories.

### 📱 Digital Harassment & Cyberstalking

* Cyberstalking
* Cyberbullying
* Online harassment
* Doxxing
* Online threats
* Online blackmailing
* Unwanted digital contact

### 🔐 Account Security & Unauthorized Access

* Account hacking
* Unauthorized access
* Unauthorized access to messages/data
* Identity theft
* Fake social-media accounts
* Impersonation
* Unauthorized SIM-related issues

### 💸 Financial Scams & Digital Fraud

* Online bank fraud
* OTP/password scams
* Phishing
* Spoofing
* Counterfeit platforms
* Spamming
* Fraudulent marketing
* Extortion
* Unauthorized data copying

### 🚨 Offences Against Person

* Physical threats
* Criminal intimidation
* Assault
* Hurt
* Rape
* Sexual offences
* Kidnapping
* Abduction
* Human trafficking
* Child-related exploitation offences

---

# 🎛️ Response Controls

Users can choose the desired response depth from the sidebar.

### Full Explanation

Provides a detailed explanation with structured sections, legal context, and practical steps.

### Main Points

Provides concise bullet points focusing on:

* Key legal information
* Penalties
* Important points
* Immediate actions

### Brief Summary

Provides a short 3–4 sentence explanation covering the main situation, legal standing, and key action steps.

---

# 💬 Example Questions

Try questions such as:

```text
What does Section 24A of PECA say about cyberbullying?

What is the punishment for cyber stalking under PECA?

What are the penalties under PPC for rape and sexual assault?

What does the Pakistan Penal Code say about criminal intimidation?

How can I report online harassment or physical threats?

Someone is blackmailing me through WhatsApp. What should I do?

Someone created a fake social media account using my name. What can I do?

Someone is threatening me online. What evidence should I preserve?
```

---

# 🏗️ Technology Stack

| Technology                | Purpose                            |
| ------------------------- | ---------------------------------- |
| Python                    | Core application development       |
| Streamlit                 | Web application and user interface |
| FAISS                     | Vector similarity search           |
| Sentence Transformers     | Legal-text embeddings              |
| `all-MiniLM-L6-v2`        | Embedding model                    |
| PyMuPDF                   | PDF text extraction                |
| NumPy                     | Vector/numerical operations        |
| Groq                      | LLM inference                      |
| `openai/gpt-oss-120b`     | Language model                     |
| GitHub                    | Source code and version control    |
| Streamlit Community Cloud | Deployment                         |

---

# 📁 Project Structure

The current application expects the legal PDFs in the **same directory as `app.py`**:

```text
CyberShield-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── PECA_2016.pdf
└── PPC_1860.pdf
```

### Important

The filenames must match the names expected by the application:

```text
PECA_2016.pdf
PPC_1860.pdf
```

The application automatically loads these files when the Streamlit application starts.

---

# ⚙️ Local Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/CyberShield-AI.git
cd CyberShield-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Add the Legal PDFs

Place these files next to `app.py`:

```text
PECA_2016.pdf
PPC_1860.pdf
```

---

## 4. Configure Groq API Key

The application looks for:

```text
GROQ_API_KEY
```

For Streamlit deployment, add the key through **Streamlit Secrets**.

Example:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

**Never commit your real API key to GitHub.**

---

## 5. Run the Application

```bash
streamlit run app.py
```

---

# ☁️ Streamlit Deployment

CyberShield AI is designed to run on Streamlit Community Cloud.

### Deployment Steps

1. Push the project to GitHub.
2. Make sure `app.py` is in the repository root.
3. Make sure the legal PDFs are also in the repository.
4. Open Streamlit Community Cloud.
5. Connect your GitHub repository.
6. Select `app.py` as the main application file.
7. Add your `GROQ_API_KEY` under Streamlit Secrets.
8. Deploy.

Your application will automatically:

```text
Load PDFs
   ↓
Extract Text
   ↓
Create Chunks
   ↓
Generate Embeddings
   ↓
Build FAISS Index
   ↓
Accept User Questions
```

---

# 🔐 Security

## API Keys

Never place API keys directly inside `app.py`.

❌ Do not use:

```python
GROQ_API_KEY = "real-api-key-here"
```

Use Streamlit Secrets instead.

### Recommended `.gitignore`

```text
.env
.streamlit/secrets.toml
__pycache__/
*.pyc
.venv/
venv/
```

---

# 🧠 Grounding & Hallucination Control

CyberShield AI uses explicit grounding instructions for legal responses.

The model is instructed to:

```text
Use ONLY the supplied document context for legal claims.
```

It is also instructed:

```text
Do NOT invent sections, penalties, procedures, or authorities.
```

If the retrieved context does not support the answer, the application returns:

> Sorry, this information is not found in the referenced legal documents (PECA / PPC).

This approach helps reduce unsupported legal claims and keeps the response tied to the application's indexed documents.

---

# 🚫 Out-of-Scope Protection

CyberShield AI includes a legal/cyber scope checker.

Questions unrelated to cyber safety, criminal offences, or Pakistani legal statutes may receive an out-of-scope response rather than being passed directly to the legal-generation pipeline.

Example:

```text
This question appears outside the scope of CyberShield & Legal AI.
Please ask a question related to cyber safety, criminal offences under PPC,
online harassment, or Pakistani legal statutes.
```

---

# ⚠️ Responsible AI

CyberShield AI is designed with a safety-first approach.

The system does not intentionally encourage:

* Hacking
* Unauthorized access
* Retaliation
* Cyberattacks
* Harassment
* Violence
* Illegal activity

For immediate physical danger, users are encouraged to prioritize personal safety and contact appropriate emergency or law-enforcement services.

---

# ⚖️ Legal Disclaimer

CyberShield & Legal AI is an **educational and informational AI application**.

It does not establish an attorney-client relationship and does not constitute professional legal advice.

Legal provisions, procedures, authorities, and contact information may change. Important legal decisions should be verified using current official Pakistani legal sources or with a qualified legal professional.

The application should not be relied upon as the sole source for legal decisions.

---

# 🚀 Future Improvements

Potential future versions could include:

* 🇵🇰 Urdu and Roman Urdu support
* 🔄 Automatic legal-document updates
* 📑 Support for additional Pakistani statutes
* 🔍 Advanced hybrid retrieval
* 🧠 Reranking models for better retrieval
* 📍 Location-aware reporting guidance
* 🏛️ Integration with official reporting channels
* 📊 Automated risk classification
* 🎯 Cybercrime issue classification
* 📎 Evidence-management assistance
* 🔐 Enhanced privacy controls
* 💬 Multilingual conversational support
* 📚 More granular legal citations
* 🧪 Automated RAG evaluation and retrieval benchmarking

---

# 🏆 Project Highlights

CyberShield AI demonstrates the practical use of:

**Generative AI + RAG + Semantic Search + Vector Retrieval + Pakistani Legal Documents + Responsible AI**

The project combines traditional document-based legal information with modern AI techniques to make complex cyber-safety and criminal-law information more accessible.

---

# 🌐 Live Application

### 🚀 CyberShield & Legal AI

https://cyber-shield-ai.streamlit.app/

---

# 👩‍💻 Author

**Minha Bibi**

Computer Science Student
AI / Machine Learning Enthusiast

---

# 📜 License & Document Usage

The application source code and included legal/reference documents may be subject to different licensing and redistribution requirements.

Before redistributing legal documents, verify that you have the appropriate right to include them in the repository.

---

### 🛡️ CyberShield & Legal AI

**Making Pakistani cyber-safety and legal information more accessible through Retrieval-Augmented Generation.**
