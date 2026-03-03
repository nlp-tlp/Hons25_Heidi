from app.execution_pages.helper import load_page, init_history, load_config, load_history, load_input
from scopes import retriever_factory

name = "concept_text"
retriever = retriever_factory("concept_descriptive")

# Page
load_page()
init_history(name)
load_config()
load_history(name)
load_input(name, retriever)
