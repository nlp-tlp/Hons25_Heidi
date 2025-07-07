import logging
import re

from llm import ChatClient
from databases import BarrickSchema
from databases import Neo4j_SKB

PROMPT_PATH = "retrievers/cypher/t2c_prompt.txt"
SCHEMA_CONTEXT = BarrickSchema.schema_to_jsonlike_str()

# Retriever
class TextToCypherRetriever:
    def __init__(self, client: ChatClient, prompt_path: str = PROMPT_PATH):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client = client
        self.neo4j_skb = Neo4j_SKB()

        with open(prompt_path) as f:
            self.prompt = f.read()

    def retrieve(self, question: str | None, extra_context: str = ""):
        if question is None:
            self.logger.info("No question given, terminating")
            return
        self.logger.info(f"Question given: {question}")

        # Get generated Cypher
        query = self.generate_cypher(question, extra_context=extra_context)
        self.logger.info(f"Generated Cypher: {query}")

        # Run command
        try:
            records = self.neo4j_skb.query(query)
            self.logger.info(f"Retrieved {len(records)} records from Neo4j.")
            return query, self.remove_ids(records), None
        except Exception as e:
            self.logger.error(f"Error running Cypher: {e}")
            return query, [], f"Error during Cypher execution: {e}"

    def generate_cypher(self, question: str, extra_context: str = ""):
        # Build prompt
        prompt = self.prompt.format(
            schema=SCHEMA_CONTEXT,
            question=question
        ) + extra_context
        self.logger.info(f"Prompting LLM using: {prompt}")

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
