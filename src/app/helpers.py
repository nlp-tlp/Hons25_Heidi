import streamlit as st

from pipeline.controller import get_chat_model_names, get_embedding_model_names

# Sidebar model selection
def sidebar_model_selection(uses_embedder: bool = False):
    st.markdown("### Model Configuration")

    # defaults
    if "active_retriever_model" not in st.session_state:
            st.session_state.active_retriever_model = "gpt-4.1"
    if "active_generator_model" not in st.session_state:
        st.session_state.active_generator_model = "gpt-4o"
    if uses_embedder and ("active_embedding_model" not in st.session_state):
            st.session_state.active_embedding_model = "text-embedding-3-small"

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

    if uses_embedder:
        st.selectbox(
            "Embedding model",
            options=get_embedding_model_names(),
            index=get_embedding_model_names().index(st.session_state.active_embedding_model),
            key="temp_embedding_model"
        )

    if st.button("Apply settings for next submit"):
        st.session_state.active_retriever_model = st.session_state.temp_retriever_model
        st.session_state.active_generator_model = st.session_state.temp_generator_model
        if uses_embedder:
            st.session_state.active_embedding_model = st.session_state.temp_embedding_model

        st.success("Settings applied successfully. These will be used on your next query.")

