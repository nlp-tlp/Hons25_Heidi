import streamlit as st
import json

from llm import chat_model_choices
from scopes import retriever_factory
from generators import FinalGenerator

retriever = retriever_factory("baseline_vectorsearch")
generator = FinalGenerator()

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

    if "active_generator_model" not in st.session_state:
        st.session_state.active_generator_model = chat_model_choices[0]

    with st.form("config_form", border=False, enter_to_submit=False):
        generator_model_config = st.selectbox(
            "Generator model",
            chat_model_choices
        )

        submitted = st.form_submit_button("Apply settings for next submit")
        if submitted:
            st.session_state["active_generator_model"] = generator_model_config

            st.success("Settings applied successfully. These will be used on your next query.")

# Display chat history
for entry in st.session_state.chat_history_vector:
    if "config" in entry:
        st.markdown(f"**Configuration:** Generator: `{entry['config']['generator_model']}`")

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
                "generator_model": st.session_state.active_generator_model,
            }

        _, results, _ = retriever.retrieve(question) # retrieval doesn't use model
        response = generator.generate(question=question, retrieved_nodes=results, schema_context=retriever.schema_context(), model=st.session_state.active_generator_model)

    st.session_state.chat_history_vector.append({"role": "user", "msg": question})
    st.session_state.chat_history_vector.append({"role": "assistant", "msg": response, "raw": results, "config": config_snapshot})
    st.rerun()
