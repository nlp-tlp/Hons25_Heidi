from app.embedding_pages.helper import load_page, init_history, load_config, load_history, load_input
from scopes import RowAllScopeGraph

graph = RowAllScopeGraph()
graph.load_chroma()
name = "row_all"

# Page
load_page()
init_history(name)
load_config(name)
load_history(name)
load_input(name, graph)