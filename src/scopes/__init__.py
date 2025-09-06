from .property_text.property_text_scope import PropertyTextScopeGraph, PropertyTextScopeRetriever
from .concept_text.concept_text_scope import ConceptTextScopeGraph, ConceptTextScopeRetriever
from .row_text.row_text_scope import RowTextScopeGraph, RowTextScopeRetriever
from .row_all.row_all_scope import RowAllScopeGraph, RowAllScopeRetriever

def retriever_factory(name: str, allow_linking: bool):
    from llm import ChatClient, EmbeddingClient

    match name:
        case "baseline_text2cypher":
            graph = PropertyTextScopeGraph()
            graph.load_neo4j()
            return PropertyTextScopeRetriever(
                graph=graph,
                prompt_path="scopes/property_text/t2c_prompt.txt",
                allow_linking=allow_linking,
                allow_extended=False,
                allow_descriptive_only=False,
                chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
                embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small")
            )
        case "property_descriptive":
            graph = PropertyTextScopeGraph()
            graph.load_neo4j()
            return PropertyTextScopeRetriever(
                graph=graph,
                prompt_path="scopes/property_text/exc_descriptive_prompt.txt",
                allow_linking=allow_linking,
                allow_extended=True,
                allow_descriptive_only=True,
                chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
                embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small")
            )
        case "property_text":
            graph = PropertyTextScopeGraph()
            graph.load_neo4j()
            return PropertyTextScopeRetriever(
                graph=graph,
                prompt_path="scopes/property_text/exc_text_prompt.txt",
                allow_linking=allow_linking,
                allow_extended=True,
                allow_descriptive_only=False,
                chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
                embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small")
            )
        case "concept_descriptive":
            graph = ConceptTextScopeGraph()
            graph.load_neo4j()
            return ConceptTextScopeRetriever(
                graph=graph,
                prompt_path="scopes/concept_text/exc_descriptive_prompt.txt",
                chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
                embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small"),
                allow_descriptive_only=True
            )
        case "concept_text":
            graph = ConceptTextScopeGraph()
            graph.load_neo4j()
            return ConceptTextScopeRetriever(
                graph=graph,
                prompt_path="scopes/concept_text/exc_text_prompt.txt",
                chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
                embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small"),
                allow_descriptive_only=False
            )
        case "row_descriptive":
            graph = RowTextScopeGraph()
            graph.load_neo4j()
            return RowTextScopeRetriever(
                graph=graph,
                prompt_path="scopes/row_text/exc_text_prompt.txt",
                chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
                embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small"),
                allow_descriptive_only=True
            )
        case "row_text":
            graph = RowTextScopeGraph()
            graph.load_neo4j()
            return RowTextScopeRetriever(
                graph=graph,
                prompt_path="scopes/row_text/exc_text_prompt.txt",
                chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
                embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small"),
                allow_descriptive_only=False
            )
        case "baseline_vectorsearch":
            graph = RowAllScopeGraph()
            graph.load_chroma()
            return RowAllScopeRetriever(
                graph=graph,
                chat_client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"),
                embedding_client=EmbeddingClient(provider="openai", model="text-embedding-3-small")
            )
        case _:
            print("Error: Not a valid retriever name.")
            return
