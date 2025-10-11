import streamlit as st
import json

from llm import ChatClient, EmbeddingClient
from scopes import RowAllScopeGraph, RowAllScopeRetriever
from generators import FinalGenerator

graph = RowAllScopeGraph()
graph.load_chroma()

retriever = RowAllScopeRetriever(
    graph=graph,
    chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
    embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small")
)
generator = FinalGenerator(client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"))

# Page
st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

st.title("Query Interface")
st.markdown("This chat runs a simple **Vector Search** strategy.")

# Chat and settings history
if "chat_history_vector" not in st.session_state:
    st.session_state.chat_history_vector = []

# Sidebar model selection
with st.sidebar:
    st.markdown("### Configuration")

    if "active_retriever_model" not in st.session_state:
        st.session_state.active_retriever_model = "gpt-4.1-2025-04-14"
    if "active_generator_model" not in st.session_state:
        st.session_state.active_generator_model = "gpt-4.1-2025-04-14"

# Display chat history
for entry in st.session_state.chat_history_vector:
    if "config" in entry:
        st.markdown(f"**Configuration:** Retriever: `{entry['config']['retriever_model']}` | Generator: `{entry['config']['generator_model']}`")

    with st.chat_message(entry["role"]):
        st.markdown(entry["msg"])

        if "raw" in entry:
            with st.expander("Show Raw Retrieved Information"):
                st.code(json.dumps(entry["raw"], indent=4), language="json", height=200)

# User input
question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config_snapshot = {
                "retriever_model": st.session_state.active_retriever_model,
                "generator_model": st.session_state.active_generator_model,
            }

        _, results, _ = retriever.retrieve(question)
        response = generator.generate(question=question, retrieved_nodes=results, schema_context=retriever.schema_context())

    st.session_state.chat_history_vector.append({"role": "user", "msg": question})
    st.session_state.chat_history_vector.append({"role": "assistant", "msg": response, "raw": results, "config": config_snapshot})
    st.rerun()
