import logging

from llm import ChatClient
from databases import BarrickSchema

# Prompts
schema_context = BarrickSchema.schema_to_jsonlike_str()

final_generator_prompt = """You are the final generator in a RAG system. The user question that has to be answered is:

{question}

Answer this question using the following already retrieved context. Speak directly to the user who asked the question and if speaking of the retrieved records, pretend like you were the one that performed the retrieval directly. You may have to infer the names of certain terms (e.g. 'fm' may represent 'failure mode'). If no records are provided in the context, do not guess and simply say so. If you think some information is missing, just assume that the retrieved information is the correct set to answer the question, but mention this in your response:

{records}

Here is some context on the way the data was stored that might be of use:

{schema}"""

# Generator
class FinalGenerator:
    def __init__(self, client: ChatClient):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = client

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
        prompt = final_generator_prompt.format(
            question=question,
            records=context_string,
            schema=schema_context
        )
        self.logger.debug(f"Prompting LLM using: {prompt}")

        # Generate final response from LLM
        final_response = self.client.chat(prompt=prompt)
        return final_response
