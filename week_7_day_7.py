readme = """
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
"""
with open("README.md", "w") as f:
  f.write(readme)
!cat README.md  # to preview

!pip install -q faiss-cpu sentence-transformers openai tiktoken

import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("🔐 Enter your OpenAI API key (hidden): ")

!pip install PyPDF2

# 1️⃣ Upload PDF Files
from google.colab import files
uploaded_files = files.upload()

# 2️⃣ Extract Text from Uploaded PDFs
from PyPDF2 import PdfReader

def extract_text_from_pdfs(uploaded):
    text_chunks = []
    for filename in uploaded:
        reader = PdfReader(filename)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
        text_chunks.extend(chunks)
    return text_chunks

documents = extract_text_from_pdfs(uploaded_files)
print(f"✅ Loaded {len(documents)} chunks from {len(uploaded_files)} PDF(s).")

# 3️⃣ Create Embeddings and Store in FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# 4️⃣ Define Retrieval Function
def retrieve_relevant_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents[i] for i in indices[0]]

# 5️⃣ GPT Answer Function (OpenAI v1.x Compatible)
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

def answer_with_context(question):
    chunks = retrieve_relevant_chunks(question, k=3)
    context = "\n".join(chunks)

    prompt = f"""
Answer the following question using ONLY the context below.

Context:
{context}

Question: {question}
Answer:
"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# 🔁 Real-time interactive user input
while True:
    user_question = input("💬 Ask a question about your PDF (or type 'exit' to stop): ")
    if user_question.lower() in ['exit', 'quit']:
        print("👋 Exiting. Thank you!")
        break
    answer = answer_with_context(user_question)
    print("🤖 Answer:", answer)
    print("-" * 60)

# ✅ Save requirements.txt with all needed packages
with open("requirements.txt", "w") as f:
    f.write("openai>=1.0.0\n")                    # OpenAI API access (GPT-3.5, GPT-4)
    f.write("PyPDF2>=3.0.1\n")                    # PDF text extraction
    f.write("faiss-cpu>=1.7.4\n")                 # FAISS for vector indexing and similarity search
    f.write("sentence-transformers>=2.2.2\n")     # Embedding model (MiniLM)
    f.write("numpy>=1.23.0\n")                    # Needed for FAISS and embedding arrays

# ✅ Display the file content
!cat requirements.txt

# ✅ Download to your computer
from google.colab import files
files.download("requirements.txt")
