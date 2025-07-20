import logging

from llm import ChatClient
from databases import BarrickSchema

# Prompts
PROMPT_PATH = "generators/generator_prompt.txt"
SCHEMA_CONTEXT = BarrickSchema.schema_to_jsonlike_str()

# Generator
class FinalGenerator:
    def __init__(self, client: ChatClient, prompt_path: str = PROMPT_PATH):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = client

        with open(prompt_path) as f:
            self.prompt = f.read()

    def generate(self, question: str, retrieved_nodes: list[dict]):
        # Cases to use prewritten answers
        if len(retrieved_nodes) == 0:
            self.logger.info("No nodes retrieved, returning pre-written response.")
            return "No records could be found. Either the answer is that there are no such entities, or that the context given was insufficient to retrieve the right records. If you believe it is the latter, try rephrasing your question."

        if len(retrieved_nodes) > 50:
            self.logger.info(f"Too many nodes retrieved: {len(retrieved_nodes)}, returning pre-written response.")
            return "Too many records were retrieved. Either the answer contains that many entities, or the model gave a bad plan of retrieval. If you believe it is the latter, try entering the question again."

        # Build prompt
        context_string = "\n".join([str(r) for r in retrieved_nodes]) if retrieved_nodes else "No relevant records found."
        prompt = self.prompt.format(
            question=question,
            records=context_string,
            schema=SCHEMA_CONTEXT
        )
        self.logger.debug(f"Prompting LLM using: {prompt}")

        # Generate final response from LLM
        final_response = self.client.chat(prompt=prompt)
        return final_response
