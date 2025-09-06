import streamlit as st
import json

from llm import ChatClient, EmbeddingClient
from scopes import PropertyTextScopeGraph, PropertyTextScopeRetriever
from generators import FinalGenerator

graph = PropertyTextScopeGraph()
graph.load_neo4j()

if "allow_linking" not in st.session_state:
    st.session_state.allow_linking = True

retriever = PropertyTextScopeRetriever(
    graph=graph,
    prompt_path="scopes/property_text/exc_descriptive_prompt.txt",
    allow_linking=st.session_state.allow_linking,
    allow_extended=True,
    allow_descriptive_only=True,
    chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
    embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small")
)
generator = FinalGenerator(client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"))

# Page
st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

st.title("Query Interface")
st.markdown("This chat runs a **Text2Cypher** strategy. Given the user's question and some appended context, the configured LLM is made to create Cypher code to execute against the existing Neo4j database. After relevant information is retrieved, they are again passed to an LLM for generating a final response. This allows the use of some custom-defined functions.")

# Chat and settings history
if "chat_history_property_descriptive" not in st.session_state:
    st.session_state.chat_history_property_descriptive = []

# Sidebar model selection
with st.sidebar:
    st.markdown("### Configuration")

    if "active_retriever_model" not in st.session_state:
        st.session_state.active_retriever_model = "gpt-4.1-2025-04-14"
    if "active_generator_model" not in st.session_state:
        st.session_state.active_generator_model = "gpt-4.1-2025-04-14"

    with st.form("config_form", border=False, enter_to_submit=False):
        linking_config = st.checkbox("Use Entity Linking", value=True)

        submitted = st.form_submit_button("Apply settings for next submit")
        if submitted:
            st.session_state.allow_linking = linking_config
            retriever.allow_linking = linking_config

            st.success("Settings applied successfully. These will be used on your next query.")

# Display chat history
for entry in st.session_state.chat_history_property_descriptive:
    if "config" in entry:
        st.markdown(f"**Configuration:** Retriever: `{entry['config']['retriever_model']}` | Generator: `{entry['config']['generator_model']}` | **Entity Linking:** `{"enabled" if entry["config"]["linking"] else "disabled"}`")

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
                "linking": st.session_state.allow_linking
            }

            cypher_query, results, error = retriever.retrieve(question)

            if error:
                response = "Error has occurred."
            else:
                linker_list = retriever.linker.linker_list_prev if st.session_state.allow_linking else ""
                response = generator.generate(question=question, retrieved_nodes=results, schema_context=retriever.schema_context(), cypher_query=cypher_query, linker_list=linker_list)

    st.session_state.chat_history_property_descriptive.append({"role": "user", "msg": question})
    if error:
        st.session_state.chat_history_property_descriptive.append({"role": "assistant", "msg": response, "cypher": cypher_query, "error": error, "config": config_snapshot})
    else:
        st.session_state.chat_history_property_descriptive.append({"role": "assistant", "msg": response, "cypher": cypher_query, "raw": results, "config": config_snapshot})
    st.rerun()