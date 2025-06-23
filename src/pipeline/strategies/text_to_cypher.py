from neo4j import GraphDatabase

import logging
import re

import os
import sys
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC_PATH)

from pipeline.llm import ChatClient
from loader.skb_barrick import BarrickSchema
from loader.skb import Neo4jSKB

# Config - change as necessary
import os
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)
logger = logging.getLogger("TextToCypher")

# Prompts
schema_context = BarrickSchema.schema_to_jsonlike_str()

text_to_cypher_prompt = """You are a system that converts natural language questions into Cypher queries.
Use the following schema to understand the Neo4j graph:

{schema}

The question you should convert is:

{question}

Only output the minimal Cypher query, with no markdown wrapping such that it can be directly executed as a query. Ensure that more than enough is retrieved - Include any information that might be useful to generate a response."""

final_generator_prompt = """You are the final generator in a RAG system. The user question that has to be answered is:

{question}

Answer this question using the following already retrieved context. Assume these are already the right answers to the question, and simply need to be put into natural language. If no records are provided in the context, do not guess and simply say so:

{records}"""

# Retriever
class TextToCypherRetriever:
    def __init__(self, client: ChatClient):
        self.neo4j_module = Neo4jSKB(uri=NEO4J_URI, auth=NEO4J_AUTH)
        self.client = client

    def retrieve(self, question: str | None):
        if question is None:
            logger.info("No question given, terminating")
            return
        logger.info(f"Question given: {question}")

        # Get generated Cypher
        query = self.generate_cypher(question)
        logger.info(f"Generated Cypher: {query}")

        # Run command
        try:
            records = self.neo4j_module.query(query)
            logger.info(f"Retrieved {len(records)} records from Neo4j.")
            return query, self.remove_ids(records), None
        except Exception as e:
            logger.error(f"Error running Cypher: {e}")
            return query, [], f"Error during Cypher execution: {e}"

    def generate_cypher(self, question: str):
        # Build prompt
        prompt = text_to_cypher_prompt.format(
            schema=schema_context,
            question=question
        )
        logger.debug(f"Prompting LLM using: {prompt}")

        # Generate Cypher from LLM
        raw_response = self.client.chat(prompt=prompt)
        cypher_query = re.sub(r"^```[a-zA-Z]*\s*|```$", "", raw_response, flags=re.MULTILINE).strip() # Remove markdown if present
        return cypher_query

    def remove_ids(self, records):
        for record in records:
            for value in record.values():
                if isinstance(value, dict) and "external_id" in value:
                    del value["external_id"]
        return records

# Example usage
# from ..final_generator import FinalGenerator

# example_question = "What components have an RPN value over 35"
# retriever = TextToCypherRetriever(model="gpt-4o")
# retrieved_records = retriever.retrieve(question=example_question)

# generator = FinalGenerator(model="gpt-4o")
# final_response = generator.generate(question=example_question, retrieved_nodes=retrieved_records)

# logger.info(f"Final response: {final_response}")