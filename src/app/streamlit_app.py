import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

pg = st.navigation({
    "Chats": [
        st.Page("chat_pages/chat_vanilla_text2cypher.py", title="Vanilla Text-to-Cypher"),
        st.Page("chat_pages/chat_property_descriptive.py", title="Property Descriptive"),
        st.Page("chat_pages/chat_property_text.py", title="Property Text"),
        st.Page("chat_pages/chat_concept_descriptive.py", title="Concept Descriptive"),
        st.Page("chat_pages/chat_concept_text.py", title="Concept Text"),
        st.Page("chat_pages/chat_row_descriptive.py", title="Row Descriptive"),
        st.Page("chat_pages/chat_row_text.py", title="Row Text"),
        st.Page("chat_pages/chat_vanilla_vectorsearch.py", title="Vanilla Vector-Search")
    ],
    "Embeddings": [
        st.Page("embedding_pages/embedding_property_text.py", title="Property Text"),
        st.Page("embedding_pages/embedding_concept_text.py", title="Concept Text"),
        st.Page("embedding_pages/embedding_row_text.py", title="Row Text"),
        st.Page("embedding_pages/embedding_row_all.py", title="Row All"),
    ],
    "Executions": [
        st.Page("execution_pages/execution_property_text.py", title="Property Text"),
        st.Page("execution_pages/execution_concept_text.py", title="Concept Text"),
        st.Page("execution_pages/execution_row_text.py", title="Row Text"),
    ]
})

pg.run()