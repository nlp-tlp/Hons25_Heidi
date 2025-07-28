import streamlit as st
import pandas as pd
# import textwrap

from app.controller import embeddings_search

st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

st.title("Vector Search Interface")
st.markdown("This chat functions as a basic search for vector embeddings using cosine similarity. The embeddings are retrieved from an existing ChromaDB vector database. Current embeddings have been generated from the `Flair` word embedding model using mean pooling to allow for sentences to be embedded.")

# Chat and settings history
if "embeddings_history_flair" not in st.session_state:
    st.session_state.embeddings_history_flair = []

# Sidebar config selection
with st.sidebar:
    st.markdown ("### Configuration")

    if "embeddings_sentence_k" not in st.session_state:
        st.session_state.embeddings_sentence_k = 25
    if "embeddings_sentence_threshold" not in st.session_state:
        st.session_state.embeddings_sentence_threshold = None

    with st.form("config_form", border=False, enter_to_submit=False):
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
            st.session_state.embeddings_sentence_k = k_config
            st.session_state.embeddings_sentence_threshold = threshold_config

            st.success("Settings applied successfully. These will be used on your next query.")

# Display chat history
role_to_icon = {
    "user": ":material/search:",
    "assistant": ":material/list:"
}

@st.cache_data
def df_to_csv(df: pd.DataFrame):
    df["Content"] = df["Content"].astype(str).str.replace('\u00A0', ' ', regex=False)
    return df.to_csv().encode("utf-8")

for i, entry in enumerate(st.session_state.embeddings_history_flair):
    with st.chat_message(entry["role"], avatar=role_to_icon[entry["role"]]):
        if "config" in entry:
            st.markdown(f"**Configuration:** K: `{entry["config"]["k"]}` | Threshold: `{entry["config"]["threshold"]}`")

        if entry["role"] == "user":
            st.markdown(entry["search"])
        else:
            df = entry["results"]
            st.table(df)
            st.download_button(
                key=i,
                label="Download as CSV",
                data=df_to_csv(df),
                file_name="flair.csv",
                mime="text/csv",
                icon=":material/download:"
            )

# User input
search = st.chat_input("Enter a search term or passage...")
if search:
    with st.chat_message("user", avatar=":material/search:"):
        st.markdown(search)

    with st.chat_message("assistant", avatar=":material/keyboard_return:"):
        with st.spinner("Searching..."):
            config_snapshot = {
                "k": st.session_state.embeddings_sentence_k,
                "threshold": st.session_state.embeddings_sentence_threshold
            }

            records = embeddings_search(
                search=search.strip(),
                embeddings_type="flair mean-pooling",
                k=st.session_state.embeddings_sentence_k,
                threshold=st.session_state.embeddings_sentence_threshold
            )

            # results = pd.DataFrame(
            #     ([record[1], "  \n".join(textwrap.wrap(record[2].replace(" ", "\u00A0"), width=100)), "{:.4f}".format(record[3])] for record in records),
            #     columns=["Type", "Content", "Score"]
            # )
            results = pd.DataFrame(
                ([record[1], record[2].replace(" ", "\u00A0"), "{:.4f}".format(record[3])] for record in records),
                columns=["Type", "Content", "Score"]
            )

            # Resolves duplicates here - TODO: Not the best way - inefficient and doesn't return same number as set 'K'
            # There being duplicates in the first place might indicate a problem with the way I am structuring data
            results = results.drop_duplicates(subset=None, keep="first", inplace=False).reset_index(drop=True)

    st.session_state.embeddings_history_flair.append({"role": "user", "search": search})
    st.session_state.embeddings_history_flair.append({"role": "assistant", "results": results, "config": config_snapshot})
    st.rerun()
