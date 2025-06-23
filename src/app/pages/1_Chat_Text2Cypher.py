import streamlit as st
import json

from app.controller import query, get_chat_model_names

st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

# Chat and settings history
if "chat_history_T2C" not in st.session_state:
    st.session_state.chat_history_T2C = []

st.title("Query Interface")
st.markdown("This chat runs a **Text2Cypher** strategy. Given the user's question and some appended context, the configured LLM is made to create Cypher code to execute against the existing Neo4j database. After relevant information is retrieved, they are again passed to an LLM for generating a final response.")

# Sidebar model selection
with st.sidebar:
    st.markdown("### Configuration")

    # defaults
    if "active_retriever_model" not in st.session_state:
            st.session_state.active_retriever_model = "gpt-4.1"
    if "active_generator_model" not in st.session_state:
        st.session_state.active_generator_model = "gpt-4o"

    # Temporary selection before save
    st.selectbox(
        "Retriever LLM",
        options=get_chat_model_names(),
        index=get_chat_model_names().index(st.session_state.active_retriever_model),
        key="temp_retriever_model"
    )

    st.selectbox(
        "Generator LLM",
        options=get_chat_model_names(),
        index=get_chat_model_names().index(st.session_state.active_generator_model),
        key="temp_generator_model"
    )

    if st.button("Apply settings for next submit"):
        st.session_state.active_retriever_model = st.session_state.temp_retriever_model
        st.session_state.active_generator_model = st.session_state.temp_generator_model

        st.success("Settings applied successfully. These will be used on your next query.")


# Display chat history
for entry in st.session_state.chat_history_T2C:
    if "config" in entry:
        st.markdown(f"**Configuration:** Retriever: `{entry['config']['retriever_model']}` | Generator: `{entry['config']['generator_model']}`")

    with st.chat_message(entry["role"]):
        st.markdown(entry["msg"])

        if "cypher" in entry and "error" in entry:
            with st.expander("Show Cypher Query"):
                st.code(entry["cypher"], language="cypher", height=200)
            with st.expander("Show Error"):
                st.code(entry["error"], wrap_lines=True, height=200)
        elif "cypher" in entry and "raw" in entry:
            with st.expander("Show Cypher Query"):
                st.code(entry["cypher"], language="cypher", height=200)
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

            cypher_query, results, response, error = query(
                question=question,
                strategy="text_to_cypher",
                retriever_model=st.session_state.active_retriever_model,
                generator_model=st.session_state.active_generator_model
            )

    st.session_state.chat_history_T2C.append({"role": "user", "msg": question})
    if error:
        st.session_state.chat_history_T2C.append({"role": "assistant", "msg": response, "cypher": cypher_query, "error": error, "config": config_snapshot})
    else:
        st.session_state.chat_history_T2C.append({"role": "assistant", "msg": response, "cypher": cypher_query, "raw": results, "config": config_snapshot})
    st.rerun()