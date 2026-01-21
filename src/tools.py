import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = PersistentClient(path="vectordb")
collection = client.get_collection("rag_chunks")

def vector_search(query: str) -> str:
    try:
        emb = embedding_model.encode([query]).tolist()
        res = collection.query(query_embeddings=emb, n_results=3) # Increased to 3
        docs = res.get("documents", [[]])[0]
        
        if not docs:
            print(f"DEBUG: No docs found for query: {query}")
            return "NO_INFO"
            
        context = " ".join(docs)
        print(f"DEBUG: Found context: {context[:100]}...") # See what was found in console
        return context
    except Exception as e:
        print(f"DEBUG: Vector Error: {e}")
        return "NO_INFO"

def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        # Return truncated text to save tokens
        return results[:500]
    except Exception as e:
        return f"Web search error: {str(e)}"        