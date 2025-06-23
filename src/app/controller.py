import yaml

import os
import sys
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC_PATH)

from pipeline.strategies.text_to_cypher import TextToCypherRetriever
from pipeline.strategies.planning_routing import PlannerRetriever
from pipeline.final_generator import FinalGenerator
from pipeline.llm import ChatClient, EmbeddingClient

# Load available models from models.yaml
chat_models = {}
embedding_models = {}
CONFIG_PATH = os.path.join(SRC_PATH, "models.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

    for model in config["chat_models"]:
        name, provider = model["name"], model["provider"]
        chat_models[name] = ChatClient(provider=provider, model=name)

    for model in config["embedding_models"]:
        name, provider, args = model["name"], model["provider"], model["options"]
        embedding_models[name] = EmbeddingClient(provider=provider, model=name, **args)

def get_chat_model_names():
    return list(chat_models.keys())

def get_embedding_model_names():
    return list(embedding_models.keys())

def rag_query(question: str, strategy: str = "text_to_cypher",
    retriever_model: str = "llama3.2", generator_model: str = "llama3.2", embedding_model: str = "text-embedding-3-small") -> str:

    match strategy:
        case "text_to_cypher":
            text_to_cypher_retriever = TextToCypherRetriever(client=chat_models[retriever_model])
            cypher_query, results, error = text_to_cypher_retriever.retrieve(question=question)
            if error:
                return cypher_query, results, "Error has occurred.", error

            final_generator = FinalGenerator(client=chat_models[generator_model])
            response = final_generator.generate(question=question, retrieved_nodes=results)
            return cypher_query, results, response, None
        case "planning_routing":
            planning_routing_retriever = PlannerRetriever(client=chat_models[retriever_model], embedding_client=embedding_models[embedding_model])
            plan, last_results, error = planning_routing_retriever.retrieve(question=question)
            if error:
                return plan, last_results, "Error has occurred.", error

            final_generator = FinalGenerator(client=chat_models[generator_model])
            response = final_generator.generate(question=question, retrieved_nodes=last_results)
            return plan, last_results, response, None
        case _:
            return "Strategy not defined. Please check configuration."

def embeddings_search(search: str, embeddings_type: str = "sentence", metric: str = "cosine"):
    pass

