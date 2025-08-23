import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

pg = st.navigation({
    "Chats": [
        # st.Page("chat_pages/chat_text2cypher_extended.py", title="Chat Text2Cypher (extended)"),
        # st.Page("chat_pages/chat_text2cypher.py", title="Chat Text2Cypher"),
        # st.Page("chat_pages/chat_text2cypher_nolink.py", title="Chat Text2Cypher (no-linking)"),
        # st.Page("chat_pages/chat_neighbourvector.py", title="Chat NeighbourVector"),
        # st.Page("chat_pages/chat_cyphervector_linear.py", title="Chat CypherVector (linear)")
        st.Page("chat_pages/chat_vanilla_text2cypher.py", title="Vanilla Text-to-Cypher"),
        st.Page("chat_pages/chat_property_descriptive.py", title="Property Descriptive"),
        st.Page("chat_pages/chat_row_text.py", title="Row Text")
    ],
    "Embeddings": [
        # st.Page("embedding_pages/embeddings_te3s.py", title="Embedding Te3s"),
        # st.Page("embedding_pages/embeddings_glove.py", title="Embedding Glove (mean-pooling)"),
        # st.Page("embedding_pages/embeddings_flair.py", title="Embedding Flair (mean-pooling)")
        st.Page("embedding_pages/embedding_property_text.py", title="Property Text")
    ],
    "Executions": [
        # st.Page("execution_pages/execution_cypher_extended.py", title="Execution Cypher (extended)")
        st.Page("execution_pages/execution_property_text.py", title="Property Text")
    ]
})

pg.run()