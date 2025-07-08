import logging
import yaml

from retrievers import TextToCypherRetriever, PlannerRetriever
from generators import FinalGenerator
from llm import ChatClient
from databases import embedder_factory
from linking import EntityLinker

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

CONFIG_PATH = "models.yaml"

# Load available models from models.yaml
chat_models = {}
embedders = {}

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

    for model in config["chat_models"]:
        name, provider = model["name"], model["provider"]
        chat_models[name] = ChatClient(provider=provider, model=name)

    for embedder in config["embedding_collections"]:
        name = embedder["name"]
        embedders[name] = embedder_factory(name=name)

linker = EntityLinker(client=ChatClient(provider="openai", model="gpt-4.1-mini-2025-04-14"))

def get_chat_model_names():
    return list(chat_models.keys())

def get_embedder_names():
    return list(embedders.keys())

def rag_query(question: str, strategy: str = "text_to_cypher",
    retriever_model: str = "llama3.2", generator_model: str = "llama3.2", embedder: str = "text-embedding-3-small") -> str:

    match strategy:
        case "text_to_cypher":
            extra_context = linker.get_linked_context(question=question)

            text_to_cypher_retriever = TextToCypherRetriever(client=chat_models[retriever_model])
            cypher_query, results, error = text_to_cypher_retriever.retrieve(question=question, extra_context=extra_context)
            if error:
                return cypher_query, results, "Error has occurred.", error

            final_generator = FinalGenerator(client=chat_models[generator_model])
            response = final_generator.generate(question=question, retrieved_nodes=results)
            return cypher_query, results, response, None
        case "planning_routing":
            extra_context = linker.get_linked_context(question=question)

            planning_routing_retriever = PlannerRetriever(client=chat_models[retriever_model], embedder=embedders[embedder])
            plan, last_results, error = planning_routing_retriever.retrieve(question=question)
            if error:
                return plan, last_results, "Error has occurred.", error

            final_generator = FinalGenerator(client=chat_models[generator_model])
            response = final_generator.generate(question=question, retrieved_nodes=last_results)
            return plan, last_results, response, None
        case _:
            return "Strategy not defined. Please check configuration."

def embeddings_search(search: str, embeddings_type: str = "text-embedding-3-small", k: int = 25, threshold: float = None):
    if embeddings_type not in get_embedder_names():
        return "Embeddings type not defined. Please check configuration."

    embedder = embedders[embeddings_type]
    return embedder.query(search, k=k, threshold=threshold)
