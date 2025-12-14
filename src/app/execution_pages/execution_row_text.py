from app.execution_pages.helper import load_page, init_history, load_config, load_history, load_input
from scopes import retriever_factory

name = "row_text"
retriever = retriever_factory("row_descriptive")

# Page
load_page()
init_history(name)
load_config()
load_history(name)
load_input(name, retriever)
