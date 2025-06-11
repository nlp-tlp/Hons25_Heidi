from .strategies.text_to_cypher import TextToCypherRetriever
from .strategies.planning_routing import PlannerRetriever
from .final_generator import FinalGenerator
from .llm import ChatClient, EmbeddingClient

gpt4o_client = ChatClient(provider="openai", model="gpt-4o")
llama32_client = ChatClient(provider="ollama", model="llama3.2")
deepseekr1_client = ChatClient(provider="ollama", model="deepseek-r1:14b")

embedding3small_embedder = EmbeddingClient(provider="openai", model="text-embedding-3-small", dimensions=256)
mxbai_embedder = EmbeddingClient(provider="ollama", model="mxbai-embed-large")

model_name_to_client = {
    "llama3.2": llama32_client,
    "gpt-4o": gpt4o_client,
    "text-embedding-3-small": embedding3small_embedder,
    "mxbai-embed-large": mxbai_embedder,
    "deepseek-r1:14b": deepseekr1_client
}

def query(question: str, strategy: str = "text_to_cypher",
    retriever_model: str = "llama3.2", generator_model: str = "llama3.2", embedding_model: str = "text-embedding-3-small") -> str:
    # Ignore the passed in schema context for now

    match strategy:
        case "text_to_cypher":
            text_to_cypher_retriever = TextToCypherRetriever(client=model_name_to_client[retriever_model])
            cypher_query, results, error = text_to_cypher_retriever.retrieve(question=question)
            if error:
                return cypher_query, results, "Error has occurred.", error

            final_generator = FinalGenerator(client=model_name_to_client[generator_model])
            response = final_generator.generate(question=question, retrieved_nodes=results)
            return cypher_query, results, response, None
        case "planning_routing":
            planning_routing_retriever = PlannerRetriever(client=model_name_to_client[retriever_model], embedding_client=model_name_to_client[embedding_model])
            plan, last_results, error = planning_routing_retriever.retrieve(question=question)
            if error:
                return plan, last_results, "Error has occurred.", error

            final_generator = FinalGenerator(client=model_name_to_client[generator_model])
            response = final_generator.generate(question=question, retrieved_nodes=last_results)
            return plan, last_results, response, None
        case _:
            return "Strategy not defined. Please check configuration."
