import logging
import re

from llm import ChatClient, EmbeddingClient
from databases import BarrickSchema
from databases import Neo4j_SKB

PROMPT_PATH = "retrievers/cypher/t2c_prompt.txt"
SCHEMA_CONTEXT = BarrickSchema.schema_to_jsonlike_str()

# Retriever
class TextToCypherRetriever:
    def __init__(self, client: ChatClient, prompt_path: str = PROMPT_PATH, embedding_client: EmbeddingClient = None):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client = client
        self.neo4j_skb = Neo4j_SKB()

        if embedding_client:
            self.embedding_client = embedding_client

        with open(prompt_path) as f:
            self.prompt = f.read()

    def retrieve(self, question: str | None, extra_context: str = "", extended_cypher: bool = False):
        if question is None:
            self.logger.info("No question given, terminating")
            return
        self.logger.info(f"Question given: {question}")

        # Get generated Cypher
        query = self.generate_cypher(question, extra_context=extra_context)
        self.logger.info(f"Generated Cypher: {query}")

        # Process extended functions and run command
        return self.execute_query(query=query, extended_cypher=extended_cypher)

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

    def execute_query(self, query: str, extended_cypher: bool = True):
        params = {}
        original_query = query
        if extended_cypher and self.embedding_client:
            query, params = self.convert_extended_functions(query)

        try:
            if params:
                records = self.neo4j_skb.query(query, other_params=params)
            else:
                records = self.neo4j_skb.query(query)
            self.logger.info(f"Retrieved {len(records)} records from Neo4j.")
            return original_query, self.remove_ids(records), None
        except Exception as e:
            self.logger.error(f"Error running Cypher: {e}")
            return original_query, [], f"Error during Cypher execution: {e}"

    def convert_extended_functions(self, query: str):
        # Fuzzy match replacement
        query = re.sub(r"IS_FUZZY_MATCH\(([^,]+),\s*([^)]+)\)", r"apoc.text.fuzzyMatch(\1, \2)", query)

        # Semantic match replacement
        match_params = re.findall(r"IS_SEMANTIC_MATCH\(([^,]+),\s*([^)]+)\)", query)
        if not match_params:
            return query, None

        params = {}
        for i, (target, search_phrase) in enumerate(match_params):
            self.logger.info(f"Processing semantic match for: {search_phrase}")
            vector = self.embedding_client.embed(search_phrase.strip())

            vector_placeholder = f"vector_{i}"
            similarity_var = f"similarity_{i}"
            target_entity = target.split('.')[0]
            query = re.sub( # TODO: Deal with OR clause
                rf"(WHERE|AND)\s+IS_SEMANTIC_MATCH\(\s*{target}\s*,\s*{search_phrase}\s*\)",
                f"WITH *, vector.similarity.cosine({target_entity}.embedding, ${vector_placeholder}) AS {similarity_var}\n"
                f"WHERE {similarity_var} > 0.665",
                query
            )
            params[vector_placeholder] = vector

        self.logger.info(f"Converted query to: {query}")
        return query, params

    def remove_ids(self, records):
        for record in records:
            for value in record.values():
                if isinstance(value, dict) and "external_id" in value:
                    del value["external_id"]
        return records
