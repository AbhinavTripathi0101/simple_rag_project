import os
import json
from src.llm import get_llm
from src.tools import vector_search, web_search

os.makedirs("memory", exist_ok=True)

def get_summary(session_id):
    path = f"memory/{session_id}_sum.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("summary", "")
    return "New conversation."

def save_summary(session_id, summary):
    with open(f"memory/{session_id}_sum.json", "w") as f:
        json.dump({"summary": summary}, f)

def run_agent(query: str, session_id="chat") -> str:
    llm = get_llm()
    summary = get_summary(session_id)


    rewrite_prompt = f"""
    Based on the Chat History, rewrite the User Question to be a standalone search query.
    Chat History: {summary}
    User Question: {query}
    Standalone Question:"""
    
    rewritten_query = llm.invoke(rewrite_prompt).content.strip()

    try:
        #  Search Documents with the REWRITTEN query
        docs_context = vector_search(rewritten_query)
        
        # Validation Logic (Keyword check)
        query_words = [w for w in rewritten_query.lower().split() if len(w) > 3]
        is_relevant = any(word in docs_context.lower() for word in query_words)

        if docs_context != "NO_INFO" and is_relevant:
            source_name = "Internal Documents"
            context = docs_context
        else:
            source_name = "Web/Live Data"
            context = web_search(rewritten_query)

        # 3. Final Synthesis
        final_prompt = f"""
        History: {summary}
        Source: {source_name}
        Context: {context}
        Question: {query}
        """
        
        response = llm.invoke(final_prompt)
        full_answer = response.content

        # UPDATED MEMORY: Save both the question AND the answer
        # This gives the agent a "rolling" memory of the conversation
        new_history = f"User asked: {query} | AI answered: {full_answer[:100]}"
        save_summary(session_id, new_history)

        return f"**[Source: {source_name}]**\n\n{full_answer}"

    except Exception as e:
        return f"Agent Error: {str(e)}"