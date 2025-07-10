import os
import re
import nltk
import fitz  # PyMuPDF
from tqdm import tqdm
from chromadb import PersistentClient
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('punkt_tab')

def read_pdf(path):
    document = fitz.open(path)
    return " ".join(page.get_text() for page in document)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return re.sub(r'\\[ntr]', ' ', text).strip()

def chunk_text(text, min_words=100, max_words=300):
    sentences = sent_tokenize(text)
    chunks, current = [], []

    for sentence in sentences:
        words = sentence.split()
        if len(current) + len(words) <= max_words:
            current.extend(words)
        else:
            if len(current) >= min_words:
                chunks.append(" ".join(current))
            current = words
    if current and len(current) >= min_words:
        chunks.append(" ".join(current))
    return chunks

def main():
    input_file = "data/AI Training Document.pdf"
    if not os.path.exists(input_file):
        raise FileNotFoundError("Expected PDF not found at data/.")
    print("[+] Reading and preprocessing document...")
    text = read_pdf(input_file)
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    print(f"[+] Generated {len(chunks)} chunks.")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    os.makedirs("chunks", exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(f"chunks/chunk_{i}.txt", "w", encoding="utf-8") as f:
            f.write(chunk)

    os.makedirs("vectordb", exist_ok=True)
    client = PersistentClient(path="vectordb")
    collection = client.get_or_create_collection("rag_chunks")
    collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=[f"chunk_{i}" for i in range(len(chunks))])
    print("[✓] Vector database built successfully.")

if __name__ == "__main__":
    main()
