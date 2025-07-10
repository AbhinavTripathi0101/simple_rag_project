import streamlit as st
from dotenv import load_dotenv
import os

from src.rag_pipeline import RAGPipeline


load_dotenv()


rag = RAGPipeline()


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title(" RAG Chatbot with Streaming ")


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown("### ℹ Chatbot Info")
    st.write(f"**Model:** {os.getenv('GROQ_MODEL')}")
    st.write(f"**Chunks in DB:** {len(rag.retriever.collection.get()['ids'])}")
    st.button(" Clear Chat", on_click=lambda: st.session_state.messages.clear())


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


user_input = st.chat_input("Ask a question...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})


    response_text = ""
    response_area = st.empty()
    response_gen, retrieved_chunks = rag.run(user_input)


    with st.expander("Retrieved Context Chunks"):
        for chunk in retrieved_chunks:
            st.markdown(f"**{chunk['id']}** (score: {chunk['distance']:.4f})")
            st.markdown(chunk["text"])
            st.markdown("---")


    for token in response_gen:
        if token:
            response_text += token
            response_area.markdown(response_text + "▌")

    response_area.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
