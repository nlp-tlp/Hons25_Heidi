import logging
import tiktoken

from llm import ChatClient

# Prompts
PROMPT_PATH = "generators/generator_prompt.txt"

# Generator
class FinalGenerator:
    def __init__(self, client: ChatClient, prompt_path: str = PROMPT_PATH):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = client

        with open(prompt_path) as f:
            self.prompt = f.read()

    def generate(self, question: str, retrieved_nodes: list[dict], schema_context: str, cypher_query: str = None, linker_list: str = None):
        # Cases to use prewritten answers
        if len(retrieved_nodes) == 0:
            self.logger.info("No nodes retrieved, returning pre-written response.")
            return "No records could be found. Either the answer is that there are no such entities, or that the context given was insufficient to retrieve the right records. If you believe it is the latter, try rephrasing your question."

        context_string = "\n".join([str(r) for r in retrieved_nodes]) if retrieved_nodes else "No relevant records found."
        enc = tiktoken.get_encoding("o200k_base") # tokeniser for gpt-4.1
        num_tokens = len(enc.encode(context_string))
        if num_tokens > 5000:
            self.logger.info(f"Too much information retrieved: {len(retrieved_nodes)} nodes with {num_tokens} tokens, returning pre-written response.")
            return "Too many records were retrieved. Either the answer contains that many entities, or the model gave a bad plan of retrieval. If you believe it is the latter, try entering the question again."

        # Build prompt
        prompt = self.prompt.format(
            question=question,
            records=context_string,
            schema=schema_context
        )
        if cypher_query:
            prompt += f"\n### Retrieval query\n\nBelow is the query that was used by your retrieval counterpart, which was instructed to give you enough context to answer the provided question:\n\n{cypher_query}\n"
        if linker_list:
            prompt += f"\n### Entity linking\n\nThis RAG strategy also involved using entity linking. This fuzzy matched list of entities was provided to the retriever before they generated the cypher query. They were instructed to pick critically.\n\nThe user asking the question may be mistaken about the right type, in which case it would be helpful to retrieve for another type that contains a very similar name. The user may also link together names that should be split across multiple entity types, in which case they were asked to check combinations of entities that make up the right names:\n\n{linker_list}"

        self.logger.info(f"Prompting LLM using: {prompt}")

        # Generate final response from LLM
        final_response = self.client.chat(prompt=prompt)
        return final_response
