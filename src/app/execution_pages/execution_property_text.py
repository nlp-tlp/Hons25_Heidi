from app.execution_pages.helper import load_page, init_history, load_config, load_history, load_input
from llm import ChatClient, EmbeddingClient
from scopes import PropertyTextScopeGraph, PropertyTextScopeRetriever

graph = PropertyTextScopeGraph()
graph.load_neo4j()
name = "property_text"

retriever = PropertyTextScopeRetriever(
    graph=graph,
    prompt_path="scopes/property_text/t2c_prompt.txt",
    allow_linking=False,
    allow_extended=True,
    allow_descriptive_only=True,
    chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
    embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small")
)

# Page
load_page()
init_history(name)
load_config()
load_history(name)
load_input(name, retriever)
