import streamlit as st
import pandas as pd

from llm import ChatClient, EmbeddingClient
from scopes import RowTextScopeGraph, RowTextScopeRetriever

graph = RowTextScopeGraph()
graph.load_neo4j()

retriever = RowTextScopeRetriever(
    graph=graph,
    prompt_path="scopes/row_text/exc_text_prompt.txt",
    chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
    embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small")
)

# Page
st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

st.title("Execution Interface")
st.markdown("This chat allows for the direct execution of Cypher queries in the form of the `Text2ExtendedCypher` chat, for testing manually written model RAG answers.")

if "execution_history_row_text" not in st.session_state:
    st.session_state.execution_history_row_text = []

# Sidebar config selection
with st.sidebar:
    st.markdown ("### Configuration")

# Display chat history
role_to_icon = {
    "user": ":material/construction:",
    "assistant": ":material/list:"
}

for entry in st.session_state.execution_history_row_text:
    with st.chat_message(entry["role"], avatar=role_to_icon[entry["role"]]):
        if entry["role"] == "user":
            st.code(entry["query"], wrap_lines=True, language="cypher")
        else:
            if "results" in entry:
                df = entry["results"]
                st.table(df)
            if "error" in entry:
                with st.expander("Show Error"):
                    st.code(entry["error"], wrap_lines=True, height=200)

# User input
query = st.chat_input("Enter a query...")
if query:
    with st.chat_message("user", avatar=":material/construction:"):
        st.code(query, wrap_lines=True, language="cypher")

    with st.chat_message("assistant", avatar=":material/list:"):
        with st.spinner("Searching..."):
            converted_query, results, error = retriever.execute_query(query=query.strip())
            results  = pd.DataFrame(results)

    st.session_state.execution_history_row_text.append({"role": "user", "query": query})
    if error:
        st.session_state.execution_history_row_text.append({"role": "assistant", "error": error})
    else:
        st.session_state.execution_history_row_text.append({"role": "assistant", "results": results})
    st.rerun()
