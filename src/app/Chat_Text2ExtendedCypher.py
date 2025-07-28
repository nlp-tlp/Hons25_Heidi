import streamlit as st
import json

from controller import rag_query, get_chat_model_names

st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

st.title("Query Interface")
st.markdown("This chat runs a **Text2Cypher** strategy. Given the user's question and some appended context, the configured LLM is made to create Cypher code to execute against the existing Neo4j database. After relevant information is retrieved, they are again passed to an LLM for generating a final response. This particular chat allows the use of some custom-defined functions.")

# Chat and settings history
if "chat_history_T2EC" not in st.session_state:
    st.session_state.chat_history_T2EC = []

# Sidebar model selection
with st.sidebar:
    st.markdown("### Configuration")

    if "active_retriever_model" not in st.session_state:
        st.session_state.active_retriever_model = "gpt-4.1"
    if "active_generator_model" not in st.session_state:
        st.session_state.active_generator_model = "gpt-4o"

    with st.form("config_form", border=False, enter_to_submit=False):
        retriever_config = st.selectbox(
            "Retriever LLM",
            options=get_chat_model_names(),
            index=get_chat_model_names().index(st.session_state.active_retriever_model)
        )

        generator_config = st.selectbox(
            "Generator LLM",
            options=get_chat_model_names(),
            index=get_chat_model_names().index(st.session_state.active_generator_model)
        )

        submitted = st.form_submit_button("Apply settings for next submit")
        if submitted:
            st.session_state.active_retriever_model = retriever_config
            st.session_state.active_generator_model = generator_config

            st.success("Settings applied successfully. These will be used on your next query.")

# Display chat history
for entry in st.session_state.chat_history_T2EC:
    if "config" in entry:
        st.markdown(f"**Configuration:** Retriever: `{entry['config']['retriever_model']}` | Generator: `{entry['config']['generator_model']}`")

    with st.chat_message(entry["role"]):
        st.markdown(entry["msg"])

        if "cypher" in entry:
            with st.expander("Show Extended Cypher Query"):
                st.code(entry["cypher"], language="cypher", height=200)

        if "error" in entry:
            with st.expander("Show Error"):
                st.code(entry["error"], wrap_lines=True, height=200)
        elif "raw" in entry:
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

            cypher_query, results, response, error = rag_query(
                question=question,
                strategy="text_to_cypher_extended",
                retriever_model=st.session_state.active_retriever_model,
                generator_model=st.session_state.active_generator_model
            )

    st.session_state.chat_history_T2EC.append({"role": "user", "msg": question})
    if error:
        st.session_state.chat_history_T2EC.append({"role": "assistant", "msg": response, "cypher": cypher_query, "error": error, "config": config_snapshot})
    else:
        st.session_state.chat_history_T2EC.append({"role": "assistant", "msg": response, "cypher": cypher_query, "raw": results, "config": config_snapshot})
    st.rerun()