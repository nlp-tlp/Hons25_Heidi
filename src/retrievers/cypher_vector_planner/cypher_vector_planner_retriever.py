import logging
from pydantic import BaseModel
from typing import Literal
import json
import re

from llm import ChatClient
from databases import BarrickSchema, Neo4j_SKB, Chroma_DB

PROMPT_PATH = "cypher_planner_prompt.txt"
SCHEMA_CONTEXT = BarrickSchema.schema_to_jsonlike_str()

# Schema constraints
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

# Individual strategies
class PlannerRetriever:
    def __init__(self, client: ChatClient, embedder: Chroma_DB, prompt_path: str = PROMPT_PATH):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client = client
        self.neo4j_skb = Neo4j_SKB()
        self.chroma_skb = embedder
        self.working_node_ids = []

        with open(prompt_path) as f:
            self.prompt = f.read()

    def retrieve(self, question: str | None):
        if question is None:
            self.logger.info("No question given, terminating")
            return
        self.logger.info(f"Question given: {question}")

        # Generate plan
        plan_text = self.generate_plan(question)

        try:
            plan = json.loads(plan_text)
        except Exception as e:
            self.logger.error(f"Error parsing plan: {e}")
            return plan_text, [], f"Error during plan parsing: {e}"

        # Loop through plan, execute KG or vector query
        try:
            for i, step in enumerate(plan["steps"]):
                is_final = i == len(plan["steps"]) - 1
                restrict = i > 0  # restrict to previous results
                self.logger.info(f"Running step {i} of plan. Final: {is_final}, Restrict: {restrict}.")

                if step["type"] == "kg":
                    records = self.kg_query(step["query"], restrict=restrict, final=is_final)
                    self.logger.info(f"Cypher query finished - Open pool of {len(records)} records from Neo4j.")

                    if len(records) == 0:
                        return plan, [], None
                elif step["type"] == "vector":
                    records = self.vector_search(step["search"], restrict=restrict, final=is_final)
                    self.logger.info(f"Vector search finished - Open pool of {len(records)} records from Chroma.")

                    if len(records) == 0:
                        return plan, [], None
                else:
                    raise ValueError(f"Unknown plan type: {step['type']}")

                if is_final:
                    return plan, records, None
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return plan, [], f"Error during plan execution: {e}"

    def generate_plan(self, question: str):
        # Build prompt
        prompt = self.prompt.format(
            schema=SCHEMA_CONTEXT,
            question=question
        )
        self.logger.debug(f"Prompting LLM using: {prompt}")

        # Generate plan and component queries from LLM
        raw_response = self.client.chat(prompt=prompt, response_format=Plan)
        plan_text = re.sub(r"^```[a-zA-Z]*\s*|```$", "", raw_response, flags=re.MULTILINE).strip() # Remove markdown if present

        self.logger.info(f"Generated Plan:\n{plan_text}")
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
        records = self.neo4j_skb.query(query, filter_ids=self.working_node_ids)

        # Clean up and store intermediate results
        if not final:
            self.working_node_ids = self.extract_element_ids(records) # Update working set only if not final
        else:
            records = self.remove_ids(records) # Remove embedding value to reduce size of final prompt

        return records

    def vector_search(self, search: str, threshold: float = None, k: int = 25, restrict: bool = False, final: bool = False):
        if not restrict:
            records = self.chroma_skb.query(query=search, threshold=threshold, k=k)
        else:
            if not self.working_node_ids:
                logging.error(f"Non-first step with no working node ids")
                return []

            logging.info(f"Running vector search:\n{search}\nwith threshold: {str(threshold)}, top-k: {str(k)}")
            records = self.chroma_skb.query(query=search, threshold=threshold, k=k, filter_ids=self.working_node_ids)

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
