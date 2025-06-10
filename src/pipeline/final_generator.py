import logging

from .llm import ChatClient

# Prompts
final_generator_prompt = """You are the final generator in a RAG system. The user question that has to be answered is:

{question}

Answer this question using the following already retrieved context. You may have to infer the names of certain terms (e.g. 'fm' may represent 'failure mode'). If no records are provided in the context, do not guess and simply say so:

{records}"""

# Config - change as necessary
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)
logger = logging.getLogger("Generator")

# Generator
class FinalGenerator:
    def __init__(self, client: ChatClient):
        self.client = client

    def generate(self, question: str, retrieved_nodes: list[dict]):
        # Build prompt
        context_string = "\n".join([str(r) for r in retrieved_nodes]) if retrieved_nodes else "No relevant records found."
        prompt = final_generator_prompt.format(
            question=question,
            records=context_string
        )
        logger.debug(f"Prompting LLM using: {prompt}")

        # Generate final response from LLM
        final_response = self.client.chat(prompt=prompt)
        return final_response