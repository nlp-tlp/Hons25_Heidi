import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.controller import query

st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

# Chat and settings history
if "chat_history_T2C" not in st.session_state:
    st.session_state.chat_history_T2C = []
if "schema_context" not in st.session_state:
    st.session_state.schema_context = "Default schema context..."  # fallback if never set

st.title("Query Interface")

# Display chat history
for entry in st.session_state.chat_history_T2C:
    with st.chat_message(entry["role"]):
        st.markdown(entry["msg"])
        if entry.get("cypher"):
            with st.expander("Show Cypher Query"):
                st.code(entry["cypher"], language="cypher")

# User input
question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            cypher_query, response = query(question, st.session_state.schema_context, strategy="text_to_cypher")

    st.session_state.chat_history_T2C.append({"role": "user", "msg": question})
    st.session_state.chat_history_T2C.append({"role": "assistant", "msg": response, "cypher": cypher_query})
    st.rerun()