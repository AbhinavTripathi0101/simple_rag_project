import os
import openai
from typing import List, Generator
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = "https://api.groq.com/openai/v1/"

class Generator:
    def __init__(self, model=os.getenv("GROQ_MODEL", "llama3-8b-8192")):
        self.model = model

    def _build_prompt(self, question: str, chunks: List[dict]) -> str:
        context = "\n\n".join([f"[{c['id']}]\n{c['text']}" for c in chunks])
        return f"""You are a helpful AI assistant. Use the provided context to answer the question accurately and concisely. Only use the context; do not make up information.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    def generate_response(self, question: str, chunks: List[dict]) -> Generator[str, None, None]:
        prompt = self._build_prompt(question, chunks)
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if hasattr(chunk.choices[0].delta, "content"):
                yield chunk.choices[0].delta.content
