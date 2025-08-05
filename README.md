
# 📄 PDF-Powered RAG Question Answering Tool (FAISS + OpenAI + Sentence Transformers)

This project allows users to **upload their own PDFs**, ask natural language questions, and get answers **based only on the uploaded content** using:

- 🧠 **OpenAI GPT-3.5/4** for answering questions with contextual prompts
- 🔍 **FAISS** for fast vector search over PDF content
- 📄 **PyPDF2** to extract text from uploaded documents
- 🧬 **Sentence Transformers** for semantic embeddings (MiniLM model)

---

## 💡 Core Features
- Upload any number of PDFs (e.g., stories, papers, research)
- Automatically extracts and chunks text
- Embeds the content and stores it in a FAISS vector index
- Answers user questions **only using content from the uploaded PDFs**
- Works in real time with a command-line or notebook input loop

---

## ⚙️ Project Components
- `PyPDF2`: Reads and extracts raw text from PDFs
- `sentence-transformers`: Embeds documents using `all-MiniLM-L6-v2`
- `faiss`: Performs fast nearest-neighbor similarity search
- `openai`: GPT model is used to generate the final answer using prompt + context

---

## 🧠 Example Use Cases
- Custom document assistants for legal, academic, or narrative PDFs
- Upload-and-ask systems for company manuals or onboarding docs
- Real-time assistants for students or researchers to study papers
- Personal memory retrievers for journals or story-based archives

---

## 🗂️ Files Overview
| File | Purpose |
|------|---------|
| `pdf_rag_tool.py` or notebook | Full pipeline from PDF upload to question answering |
| `forest_of_echoes.pdf` | Example story PDF used for testing |
| `README.md` | You’re reading it! |
| `requirements.txt` | All necessary dependencies |

---

## ▶️ Run the Tool (in Colab or Locally)
```bash
# Step 1: Install dependencies
pip install PyPDF2 sentence-transformers faiss-cpu openai

# Step 2: Set your API key
import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = getpass("🔐 Enter your OpenAI API key (hidden): ")

# Step 3: Run the code blocks to upload PDFs, extract text, and query
