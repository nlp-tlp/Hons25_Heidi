import streamlit as st
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.controller import query

st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

# Chat and settings history
if "chat_history_planner" not in st.session_state:
    st.session_state.chat_history_planner = []
if "schema_context" not in st.session_state:
    st.session_state.schema_context = "Default schema context..."  # fallback if never set

st.title("Query Interface")

# Display chat history
for entry in st.session_state.chat_history_planner:
    with st.chat_message(entry["role"]):
        st.markdown(entry["msg"])
        if entry.get("plan"):
            with st.expander("Show Plan and Queries"):
                st.code(json.dumps(entry["plan"], indent=4), language="json", height=200)
        if entry.get("raw"):
            with st.expander("Show Raw Retrieved Information"):
                st.code(json.dumps(entry["raw"], indent=4), language="json", height=200)

# User input
question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            plan, last_results, response = query(question, st.session_state.schema_context, strategy="planning_routing")

    st.session_state.chat_history_planner.append({"role": "user", "msg": question})
    st.session_state.chat_history_planner.append({"role": "assistant", "msg": response, "plan": plan, "raw": last_results})
    st.rerun()