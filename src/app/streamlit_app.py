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
        st.Page("chat_pages/chat_row_text.py", title="Row Text")
    ],
    "Embeddings": [
        st.Page("embedding_pages/embedding_property_text.py", title="Property Text"),
        st.Page("embedding_pages/embedding_row_text.py", title="Row Text")
    ],
    "Executions": [
        st.Page("execution_pages/execution_property_text.py", title="Property Text"),
        st.Page("execution_pages/execution_row_text.py", title="Row Text"),
    ]
})

pg.run()