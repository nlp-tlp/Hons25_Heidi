import streamlit as st
import pandas as pd
import textwrap

from app.controller import embeddings_search

st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

st.title("Vector Search Interface")
st.markdown("This chat functions as a basic search for vector similarity calculated from **Sentence Embedding** models. The similar embeddings are retrieved from an existing vector database. Current embeddings have been generated from the OpenAI `text-embedding-3-small` model.")

# Chat and settings history
if "embeddings_history_sentence" not in st.session_state:
    st.session_state.embeddings_history_sentence = []

# Sidebar config selection
with st.sidebar:
    st.markdown ("### Configuration")

    if "active_embeddings_model" not in st.session_state:
        st.session_state.embeddings_sentence_metric = "cosine"
    if "embeddings_sentence_k" not in st.session_state:
        st.session_state.embeddings_sentence_k = 25
    if "embeddings_sentence_threshold" not in st.session_state:
        st.session_state.embeddings_sentence_threshold = None

    with st.form("config_form", border=False, enter_to_submit=False):
        metric_config = st.selectbox(
            "Similarity metric",
            options=["cosine"],
            index=0
        )

        k_config = st.number_input(
            "Top-K",
            min_value=1,
            value=25,
        )

        threshold_config = st.number_input(
            "Score threshold",
            min_value=float(0),
            max_value=float(1),
            step=0.01,
            value=None,
            placeholder="None",
        )

        submitted = st.form_submit_button("Apply settings for next submit")
        if submitted:
            st.session_state.embeddings_sentence_metric = metric_config
            st.session_state.embeddings_sentence_k = k_config
            st.session_state.embeddings_sentence_threshold = threshold_config

            st.success("Settings applied successfully. These will be used on your next query.")

# Display chat history
role_to_icon = {
    "user": ":material/search:",
    "assistant": ":material/list:"
}

for entry in st.session_state.embeddings_history_sentence:
    with st.chat_message(entry["role"], avatar=role_to_icon[entry["role"]]):
        if "config" in entry:
            st.markdown(f"**Configuration:** Metric: `{entry["config"]["metric"]}` | K: `{entry["config"]["k"]}` | Threshold: `{entry["config"]["threshold"]}`")

        if entry["role"] == "user":
            st.markdown(entry["search"])
        else:
            st.table(entry["results"])

# User input
search = st.chat_input("Enter a search term or passage...")
if search:
    with st.chat_message("user", avatar=":material/search:"):
        st.markdown(search)

    with st.chat_message("assistant", avatar=":material/keyboard_return:"):
        with st.spinner("Searching..."):
            config_snapshot = {
                "metric": st.session_state.embeddings_sentence_metric,
                "k": st.session_state.embeddings_sentence_k,
                "threshold": st.session_state.embeddings_sentence_threshold
            }

            records = embeddings_search(
                search=search,
                embeddings_type="sentence",
                metric=st.session_state.embeddings_sentence_metric,
                k=st.session_state.embeddings_sentence_k,
                threshold=st.session_state.embeddings_sentence_threshold
            )

            results = pd.DataFrame(([record[1], "  \n".join(textwrap.wrap(record[2], width=110)), "{:.4f}".format(record[3])] for record in records),
            columns=["Type", "Content", "Score"])

    st.session_state.embeddings_history_sentence.append({"role": "user", "search": search})
    st.session_state.embeddings_history_sentence.append({"role": "assistant", "results": results, "config": config_snapshot})
    st.rerun()
