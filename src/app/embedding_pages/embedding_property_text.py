from app.embedding_pages.helper import load_page, init_history, load_config, load_history, load_input
from scopes import PropertyTextScopeGraph

graph = PropertyTextScopeGraph()
graph.load_chroma()
name = "property_text"

# Page
load_page()
init_history(name)
load_config(name)
load_history(name)
load_input(name, graph)