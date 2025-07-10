from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, db_path="vectordb", collection_name="rag_chunks", top_k=5):
        self.top_k = top_k
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = PersistentClient(path=db_path)
        self.collection = self._load_collection(collection_name)

    def _load_collection(self, name):
        available = [col.name for col in self.client.list_collections()]
        if name in available:
            return self.client.get_collection(name=name)
        print(f"[!] Collection '{name}' not found. Creating new one.")
        return self.client.create_collection(name=name)

    def query(self, question):
        embedding = self.model.encode([question])[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.top_k,
            include=["documents", "distances"]
        )
        return [
            {"id": f"chunk_{i}", "text": doc, "distance": dist}
            for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0]))
        ]
