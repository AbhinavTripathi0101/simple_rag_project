import streamlit as st
from src.agent import run_agent
import os

st.set_page_config(page_title="Stable Agentic RAG", layout="centered")
st.title("🤖  Agentic RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar cleanup
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    if os.path.exists("memory/chat_sum.json"):
        os.remove("memory/chat_sum.json")
    st.rerun()

# Display Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask about your PDF or general info...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            answer = run_agent(user_input)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})