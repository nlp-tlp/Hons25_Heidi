import logging
import json

from llm import ChatClient
from databases import Neo4j_DB

LINKER_PROMPT_PATH = "linking/linker_prompt.txt"
RETRIEVAL_PROMPT_EXTENSION = "linking/retrieval_prompt_extension.txt"

# Linker
class EntityLinker:
    def __init__(self, client: ChatClient, graph: Neo4j_DB, prompt_path: str = LINKER_PROMPT_PATH, retrieval_prompt_ex_path: str = RETRIEVAL_PROMPT_EXTENSION):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client = client
        self.graph = graph

        with open(prompt_path) as f:
            self.prompt = f.read()

        with open(retrieval_prompt_ex_path) as f:
            self.retrieval_prompt_extension = f.read()

    def extract(self, question: str):
        prompt = self.prompt.format(
            phrase=question
        )
        self.logger.debug(f"Prompting entity extraction LLM using {prompt}")

        response = self.client.chat(prompt=prompt)
        self.logger.info(f"Retrieved raw response from LLM: {response}")

        return json.loads(response)

    def fuzzy_search(self, phrases: list[str]):
        if not phrases:
            return []

        matches = ""
        for phrase in phrases:
            matches += f"For '{phrase}':\n" + "\n".join(str(m) for m in self.graph.neo4j.ftsearch(phrase)) + "\n\n"

        return matches

    def get_linked_context(self, question: str):
        extraction = self.extract(question)
        extraction = [e.replace("(", "").replace(")", "") for e in extraction]
        matches = self.fuzzy_search(extraction)

        return f"\n{self.retrieval_prompt_extension}\n\n{matches}"