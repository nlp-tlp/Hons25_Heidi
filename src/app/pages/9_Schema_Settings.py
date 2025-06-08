import streamlit as st

st.title("Additional Context Settings")

if "schema_context" not in st.session_state:
    st.session_state.schema_context = ""

schema = st.text_area("Context to append to every prompt:", value=st.session_state.schema_context, height=300)

if st.button("Update Schema Context"):
    st.session_state.schema_context = schema
    st.success("Schema updated successfully.")
