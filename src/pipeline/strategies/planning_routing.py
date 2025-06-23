from neo4j import GraphDatabase
from pydantic import BaseModel
from typing import Literal

import logging
import json
import re

import os
import sys
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC_PATH)

from pipeline.llm import ChatClient, EmbeddingClient
from loader.skb_barrick import BarrickSchema
from loader.skb import Neo4jSKB, ChromaSKB

# Config - change as necessary
import os
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

logging.basicConfig(
    level=logging.DEBUG,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)
logger = logging.getLogger("Agentic")

# Prompts and schemas
class KGStep(BaseModel):
    type: Literal["kg"]
    query: str

class VectorStep(BaseModel):
    type: Literal["vector"]
    search: str

class ReasoningMixin:
    reasoning: str

class Plan(BaseModel, ReasoningMixin):
    steps: list[KGStep | VectorStep]

schema_context = BarrickSchema.schema_to_jsonlike_str()

planning_prompt = """You are a system that converts natural language questions into retrieval plans. There are two possible modes of retrieval - vector embedding search and KG Cypher queries. Use the following schema and information to understand the Neo4j graph:

{schema}

The question you should convert is:

{question}

Decompose this question into components that work best with structured KG search and unstructured semantic search. Each step should be used to narrow down the possible nodes and solve subproblems, and are not separate to each other but are used as a sequential retrieval process. Output in exactly the following format:

{{
    "steps": [
        {{"type": "kg", "query": "<Cypher query string>"}},
        {{"type": "vector", "search": "<semantic search string>"}},
        ...
    ],
    "reasoning": "<thought process>"
}}

Only decompose if needed - often a single step is sufficient. For each KG query component that is not the final one, end the query with `RETURN DISTINCT n.external_id AS id`. You may include any reasoning or thought processes only as part of the "reasoning" property. Do not use "$" variables or any other undefined variables.

If not the first retrieval, the nodes from the previous result is stored in the variable "n" that should be used (as full nodes and not a list, so do not add a filter clause for id filtering or use something like "IN"). Assume that "n" will have the same entity type as the entity in the last "kg" return, with the relationships that exist for that entity in the provided schema. Ensure that subsequent queries take into account this input type, and that Cypher queries do not use the wrong properties for the wrong type.

Vector searches will only return the node id, and in the same node type it was given. Properties tagged with '@match_semantically' can be searched this way. The final step should return all relevant properties needed to address the original question (instead of full nodes, e.g. RETURN fm.name, fm.rpn, ... instead of fm), and properties need to be reretrieved via Cypher if necessary. More properties can be returned than necessary if they would make for a more informative natural language answer."""

# Individual strategies
class PlannerRetriever:
    def __init__(self, client: ChatClient, embedding_client: EmbeddingClient):
        self.neo4j_module = Neo4jSKB(uri=NEO4J_URI, auth=NEO4J_AUTH)
        self.chroma_module = ChromaSKB(persist_directory=CHROMA_DB_PATH)
        self.client = client
        self.embedding_client = embedding_client

        self.working_node_ids = []
        self.chroma_module.load()

    def retrieve(self, question: str | None):
        if question is None:
            logger.info("No question given, terminating")
            return
        logger.info(f"Question given: {question}")

        # Generate plan
        plan_text = self.generate_plan(question)

        try:
            plan = json.loads(plan_text)
        except Exception as e:
            logger.error(f"Error parsing plan: {e}")
            return plan_text, [], f"Error during plan parsing: {e}"

        # Loop through plan, execute KG or vector query
        try:
            for i, step in enumerate(plan["steps"]):
                is_final = i == len(plan["steps"]) - 1
                restrict = i > 0  # restrict to previous results
                logger.info(f"Running step {i} of plan. Final: {is_final}, Restrict: {restrict}.")

                if step["type"] == "kg":
                    records = self.kg_query(step["query"], restrict=restrict, final=is_final)
                    logger.info(f"Cypher query finished - Open pool of {len(records)} records from Neo4j.")

                    if len(records) == 0:
                        return plan, [], None
                elif step["type"] == "vector":
                    records = self.vector_search(step["search"], restrict=restrict, final=is_final)
                    logger.info(f"Vector search finished - Open pool of {len(records)} records from Chroma.")

                    if len(records) == 0:
                        return plan, [], None
                else:
                    raise ValueError(f"Unknown plan type: {step['type']}")

                if is_final:
                    return plan, records, None
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return plan, [], f"Error during plan execution: {e}"

    def generate_plan(self, question: str):
        # Build prompt
        prompt = planning_prompt.format(
            schema=schema_context,
            question=question
        )
        logger.debug(f"Prompting LLM using: {prompt}")

        # Generate plan and component queries from LLM
        raw_response = self.client.chat(prompt=prompt, response_format=Plan)
        plan_text = re.sub(r"^```[a-zA-Z]*\s*|```$", "", raw_response, flags=re.MULTILINE).strip() # Remove markdown if present

        logger.info(f"Generated Plan:\n{plan_text}")
        return plan_text

    def kg_query(self, query: str, restrict=False, final=False):
        if restrict and self.working_node_ids:
            query = f"""
            UNWIND $ids AS id
            MATCH (n) WHERE n.external_id = id
            WITH DISTINCT n
            {query}
            """

        logging.info(f"Running Cypher query:\n{query}")
        records = self.neo4j_module.query(query, filter_ids=self.working_node_ids)

        # Clean up and store intermediate results
        if not final:
            self.working_node_ids = self.extract_element_ids(records) # Update working set only if not final
        else:
            records = self.remove_ids(records) # Remove embedding value to reduce size of final prompt

        return records

    def vector_search(self, search: str, threshold: float = None, k: int = None, restrict: bool = False, final: bool = False):
        if not restrict:
            records = self.chroma_module.similarity_search(search_string=search, threshold=threshold, k=k)
        else:
            if not self.working_node_ids:
                logging.error(f"Non-first step with no working node ids")
                return []

            if not threshold and not k:
                threshold = 0.3

            logging.info(f"Running vector search:\n{search}\nwith threshold: {str(threshold)}, top-k: {str(k)}")
            records = self.chroma_module.similarity_search(search_string=search, threshold=threshold, k=k, filter_ids=self.working_node_ids)

        # Clean up and store intermediate results
        if not final:
            self.working_node_ids = [record[0] for record in records]

        return records

    def extract_element_ids(self, records):
        ids = []
        for record in records:
            id = record.get("id")
            if id:
                ids.append(id)
        return ids

    def remove_ids(self, records):
        for record in records:
            for value in record.values():
                if isinstance(value, dict) and "external_id" in value:
                    del value["external_id"]
        return records

# Example usage
# example_question = "In the Fuel System component, what are the failure modes that are associated with temperature issues as a failure effect (temperature issues in failure effect node not failure mode)? Let me know what the associated failure effects were as well along with the failure modes."
# retriever = PlannerRetriever(model="gpt-4o")
# retrieved_records = retriever.retrieve(question=example_question)

# generator = FinalGenerator(model="gpt-4o")
# final_response = generator.generate(question=example_question, retrieved_nodes=retrieved_records)

# logger.info(f"Final response: {final_response}")
