from src.retriever import Retriever
from src.generator import Generator

class RAGPipeline:
    def __init__(self, top_k=5):
        self.retriever = Retriever(top_k=top_k)
        self.generator = Generator()

    def run(self, question: str):
        chunks = self.retriever.query(question)
        response_stream = self.generator.generate_response(question, chunks)
        return response_stream, chunks
