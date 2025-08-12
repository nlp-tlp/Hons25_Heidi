import logging
from pydantic import BaseModel
from typing import Literal
import json
import re

from llm import ChatClient, EmbeddingClient
from databases import BarrickSchema, Neo4j_SKB

PROMPT_PATH = "retrievers/neighbour_vector/neighbour_vector_prompt.txt"
SCHEMA_CONTEXT = BarrickSchema.schema_to_jsonlike_str()

# Schema constraints
class QueryStep(BaseModel):
    type: Literal["query"]
    query: str

class SearchStep(BaseModel):
    type: Literal["search"]
    search: str

class ReasoningMixin:
    reasoning: str

class Plan(BaseModel, ReasoningMixin):
    steps: list[QueryStep | SearchStep]

# Individual strategy
class NeighbourVectorRetriever:
    def __init__(self, client: ChatClient, prompt_path: str = PROMPT_PATH, embedding_client: EmbeddingClient = None):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client = client
        self.embedding_client = embedding_client
        self.neo4j_skb = Neo4j_SKB()

        with open(prompt_path) as f:
            self.prompt = f.read()

    def retrieve(self, question: str | None, extra_context: str = ""):
        if question is None:
            self.logger.info("No question given, terminating")
            return
        self.logger.info(f"Question given: {question}")

        # Generate plan
        plan_text = self.generate_plan(question, extra_context=extra_context)

        try:
            plan = json.loads(plan_text)
        except Exception as e:
            self.logger.error(f"Error parsing plan: {e}")
            return plan_text, [], f"Error during plan parsing: {e}"

        # Loop through plan to build Cypher query and get embeddings
        try:
            embeddings = []
            for i, step in enumerate(plan["steps"]):
                phrase = step.get("query") or step.get("search")
                if not phrase:
                    raise ValueError(f"No query or search phrase in step {i}")
                embedding = self.embedding_client.embed(phrase)
                embeddings.append(embedding)
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            return plan, [], f"Error during plan processing: {e}"

        # Execute cypher query
        try:
            cypher_query, emb_vars = self.generate_cypher_from_plan(plan, top_k=5)
            parameters = {ev: e for ev, e in zip(emb_vars, embeddings)}

            self.logger.info(f"Generated Cypher:\n{cypher_query}")

            records = self.neo4j_skb.query(cypher_query, other_params=parameters)
            self.logger.info(f"Number of triples retrieved: {len(records[0]["final_pool"])}")
            return plan, records, None
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return plan, [], f"Error during plan execution: {e}"

    def generate_plan(self, question: str, extra_context: str = ""):
        # Build prompt
        prompt = self.prompt.format(
            schema=SCHEMA_CONTEXT,
            question=question
        ) + extra_context
        self.logger.debug(f"Prompting LLM using: {prompt}")

        # Generate plan and component queries from LLM
        raw_response = self.client.chat(prompt=prompt, response_format=Plan)
        plan_text = re.sub(r"^```[a-zA-Z]*\s*|```$", "", raw_response, flags=re.MULTILINE).strip() # Remove markdown if present

        self.logger.info(f"Generated Plan:\n{plan_text}")
        return plan_text

    def generate_cypher_from_plan(self, plan: dict, top_k: int = 5):
        cypher_lines = []
        emb_vars = []

        for i, step in enumerate(plan["steps"]):
            emb_var = f"embedding{i}"
            emb_vars.append(emb_var)

            if step["type"] == "query":
                cypher_lines.append(f"""\
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                WITH n, vector.similarity.cosine(n.embedding, ${emb_var}) AS score{i}
                ORDER BY score{i} DESC
                LIMIT {top_k}
                WITH collect(n) AS intermediate_set{", [] AS final_pool" if i == 0 else ", final_pool"}
                """)

            elif step["type"] == "search":
                prev = "intermediate_set"
                cypher_lines.append(f"""\
                UNWIND {prev} AS iset{i}
                MATCH (iset{i})-[r{i}]-(neighbor{i})
                WITH DISTINCT iset{i}, neighbor{i}, type(r{i}) AS rel_type{i}, r{i}, {prev}, final_pool
                WITH iset{i}, neighbor{i}, rel_type{i}, vector.similarity.cosine(neighbor{i}.embedding, ${emb_var}) AS score{i}, {prev}, final_pool
                ORDER BY score{i} DESC
                LIMIT {top_k}
                WITH collect({{
                    source: apoc.map.removeKey(apoc.map.removeKey(iset{i}, 'embedding'), 'external_id'),
                    rel: rel_type{i},
                    target: apoc.map.removeKey(apoc.map.removeKey(neighbor{i}, 'embedding'), 'external_id')
                }}) AS new_triples, {prev}, final_pool
                WITH apoc.coll.toSet(final_pool + new_triples) AS final_pool,
                    apoc.coll.toSet({prev}) AS intermediate_set
                """)
            else:
                raise ValueError(f"Unknown step type: {step['type']}")

        cypher_lines.append("RETURN final_pool")
        return "\n".join(cypher_lines), emb_vars
