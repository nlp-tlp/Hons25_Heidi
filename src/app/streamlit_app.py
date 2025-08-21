import streamlit as st

pg = st.navigation({
    "Chats": [
        st.Page("chat_pages/chat_text2cypher_extended.py", title="Chat Text2Cypher (extended)"),
        st.Page("chat_pages/chat_text2cypher.py", title="Chat Text2Cypher"),
        st.Page("chat_pages/chat_text2cypher_nolink.py", title="Chat Text2Cypher (no-linking)"),
        st.Page("chat_pages/chat_neighbourvector.py", title="Chat NeighbourVector"),
        st.Page("chat_pages/chat_cyphervector_linear.py", title="Chat CypherVector (linear)")
    ],
    "Embeddings": [
        st.Page("embedding_pages/embeddings_te3s.py", title="Embedding Te3s"),
        st.Page("embedding_pages/embeddings_glove.py", title="Embedding Glove (mean-pooling)"),
        st.Page("embedding_pages/embeddings_flair.py", title="Embedding Flair (mean-pooling)")
    ],
    "Executions": [
        st.Page("execution_pages/execution_cypher_extended.py", title="Execution Cypher (extended)")
    ]
})

pg.run()