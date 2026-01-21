import os
import re
import nltk
import fitz
from chromadb import PersistentClient
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

def read_pdf(path):
    doc = fitz.open(path)
    return " ".join(page.get_text() for page in doc)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, min_words=80, max_words=180):
    sentences = sent_tokenize(text)
    chunks, current = [], []

    for sent in sentences:
        words = sent.split()
        if len(current) + len(words) <= max_words:
            current.extend(words)
        else:
            if len(current) >= min_words:
                chunks.append(" ".join(current))
            current = words

    if len(current) >= min_words:
        chunks.append(" ".join(current))

    return chunks

def main():
    pdf_path = "data/AI Training Document.pdf"
    text = clean_text(read_pdf(pdf_path))
    chunks = chunk_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    client = PersistentClient(path="vectordb")
    collection = client.get_or_create_collection("rag_chunks")

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f"[✓] Stored {len(chunks)} chunks in ChromaDB")

if __name__ == "__main__":
    main()
