from .strategies.text_to_cypher import TextToCypherRetriever
from .strategies.planning_routing import PlannerRetriever
from .final_generator import FinalGenerator

generator = FinalGenerator()
text_to_cypher_retriever = TextToCypherRetriever()
planning_routing_retriever = PlannerRetriever()

def query(question: str, schema: str, strategy="text_to_cypher") -> str:
    # Ignore the passed in schema for now

    match strategy:
        case "text_to_cypher":
            cypher_query, results = text_to_cypher_retriever.retrieve(question=question)
            response = generator.generate(question=question, retrieved_nodes=results)
            return cypher_query, response
        case "planning_routing":
            plan, last_results = planning_routing_retriever.retrieve(question=question)
            response = generator.generate(question=question, retrieved_nodes=last_results)
            return plan, last_results, response
        case _:
            return "Strategy not defined. Please check configuration."
