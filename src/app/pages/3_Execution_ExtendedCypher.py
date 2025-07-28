import streamlit as st
import pandas as pd
# import textwrap

from app.controller import manual_query

st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

st.title("Execution Interface")
st.markdown("This chat allows for the direct execution of Cypher queries in the form of the `Text2ExtendedCypher` chat, for testing manually written model RAG answers.")

# Chat and settings history
if "execution_history_ecypher" not in st.session_state:
    st.session_state.execution_history_ecypher = []

# Sidebar config selection
with st.sidebar:
    st.markdown ("### Configuration")

# Display chat history
role_to_icon = {
    "user": ":material/construction:",
    "assistant": ":material/list:"
}

for entry in st.session_state.execution_history_ecypher:
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
            converted_query, results, error = manual_query(question=query.strip())
            results  = pd.DataFrame(results)

    st.session_state.execution_history_ecypher.append({"role": "user", "query": query})
    if error:
        st.session_state.execution_history_ecypher.append({"role": "assistant", "error": error})
    else:
        st.session_state.execution_history_ecypher.append({"role": "assistant", "results": results})
    st.rerun()
