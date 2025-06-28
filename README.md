# Honours - Heidi Leow

Common commands have been put into a makefile. In the root directory, use `make help` to see a list of the commands and their descriptions.

## Loading

🚧 _Currently manually done_ 🚧

## Demo

### Streamlit interface

A Streamlit chat interface is available for both the current RAG strategies and vector search collections. To see the demo, run `make run_app`.

#### Issues

Using Streamlit and pyTorch (used in Flair embeddings) together throws an error when the Streamlit interface is first run. This does not impact any operations. The fix unmerged in https://github.com/streamlit/streamlit/pull/10388; adding the proposed code stops it from coming up in terminal.
