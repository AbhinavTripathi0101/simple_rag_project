# 🤖 RAG PDF Chatbot (Groq + ChromaDB + Streamlit)

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on a provided PDF. It uses sentence-transformer embeddings, ChromaDB for vector storage, and Groq-hosted LLMs (like LLaMA 3 or Mixtral) via the OpenAI-compatible API for fast, streaming responses.

---

## 🧠 Project Architecture

PDF → Chunking → Embeddings → ChromaDB
↓
Query → Retriever
↓
Context → LLM (Groq)
↓
Final Answer (Streamed)


- **Preprocessing:** Extracts, cleans, and chunks the PDF
- **Embedding:** Chunks embedded via `all-MiniLM-L6-v2` model
- **Storage:** Chunks + embeddings stored in ChromaDB
- **Retrieval:** Relevant chunks retrieved based on user query
- **Generation:** Prompt + context sent to Groq-hosted LLM (`llama3-8b-8192`)
- **Streaming:** Real-time response via Streamlit UI

---

## ⚙️ Setup Instructions

1. **Clone the repo and install dependencies**
   ```bash
   git clone <repo>
   cd project/
   pip install -r requirements.txt

2. **Set up .env**
    GROQ_API_KEY=your_actual_groq_key
GROQ_MODEL=llama3-8b-8192


3. **Place your PDF**:
    data/AI Training Document.pdf


4. 4.**Run preprocessing + embedding**-
    python notebooks/preprocess_and_embed.py

##  💬 Run the Chatbot (Streaming UI)

streamlit run app.py

## 🔍 Model & Embedding Choices

Embedding Model: all-MiniLM-L6-v2 (fast, compact, high-quality sentence embeddings)

LLM (via Groq): llama3-8b-8192 (low-latency, Groq-optimized, OpenAI-compatible)

## 🎥 Demo Preview
 Demo Video Link-https://www.loom.com/share/7dca7ca2d6674345922b21ae272c384e?sid=c5b0d7b4-fea7-414e-96ea-cfc341fb0df0
 
## ✅ Credits
Developed by Abhinav Tripathi