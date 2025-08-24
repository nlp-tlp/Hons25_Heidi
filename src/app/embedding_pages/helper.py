import streamlit as st
import pandas as pd

def load_page():
    st.set_page_config(page_title="Semi-Structured RAG demo", layout="wide")

    st.title("Vector Search Interface")
    st.markdown("This chat functions as a basic search for vector embeddings using cosine similarity. The embeddings are retrieved from an existing ChromaDB vector database. Current embeddings have been generated from the OpenAI `text-embedding-3-small` model.")

def init_history(name):
    if f"embeddings_history_{name}" not in st.session_state:
        st.session_state[f"embeddings_history_{name}"] = []

def load_config(name):
    with st.sidebar:
        st.markdown ("### Configuration")

        if f"embeddings_{name}_k" not in st.session_state:
            st.session_state[f"embeddings_{name}_k"] = 25
        if f"embeddings_{name}_threshold" not in st.session_state:
            st.session_state[f"embeddings_{name}_threshold"] = None

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
                st.session_state[f"embeddings_{name}_k"] = k_config
                st.session_state[f"embeddings_{name}_threshold"] = threshold_config

                st.success("Settings applied successfully. These will be used on your next query.")

def load_history(name):
    role_to_icon = {
        "user": ":material/search:",
        "assistant": ":material/list:"
    }

    @st.cache_data
    def df_to_csv(df: pd.DataFrame):
        df["Content"] = df["Content"].astype(str).str.replace('\u00A0', ' ', regex=False)
        return df.to_csv().encode("utf-8")

    for i, entry in enumerate(st.session_state[f"embeddings_history_{name}"]):
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
                    file_name=f"{name}.csv",
                    mime="text/csv",
                    icon=":material/download:"
                )

def load_input(name, graph):
    search = st.chat_input("Enter a search term or passage...")
    if search:
        with st.chat_message("user", avatar=":material/search:"):
            st.markdown(search)

        with st.chat_message("assistant", avatar=":material/keyboard_return:"):
            with st.spinner("Searching..."):
                config_snapshot = {
                    "k": st.session_state[f"embeddings_{name}_k"],
                    "threshold": st.session_state[f"embeddings_{name}_threshold"]
                }

                records = graph.chroma.query(
                    query=search.strip(),
                    k=st.session_state[f"embeddings_{name}_k"],
                    threshold=st.session_state[f"embeddings_{name}_threshold"]
                )

                results = pd.DataFrame(
                    ([record[1], record[2].replace(" ", "\u00A0"), "{:.4f}".format(record[3])] for record in records),
                    columns=["Type", "Content", "Score"]
                )

        st.session_state[f"embeddings_history_{name}"].append({"role": "user", "search": search})
        st.session_state[f"embeddings_history_{name}"].append({"role": "assistant", "results": results, "config": config_snapshot})
        st.rerun()
