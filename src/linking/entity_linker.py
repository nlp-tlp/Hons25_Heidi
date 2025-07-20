import logging
import json

from llm import ChatClient
from databases import Fuzzy_SKB

PROMPT_PATH = "linking/linker_prompt.txt"

# Linker
class EntityLinker:
    def __init__(self, client: ChatClient, prompt_path: str = PROMPT_PATH):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client = client
        self.fuzzy_skb = Fuzzy_SKB()
        self.fuzzy_skb.load_pickle()

        with open(prompt_path) as f:
            self.prompt = f.read()

    def extract(self, question: str):
        prompt = self.prompt.format(
            phrase=question
        )
        self.logger.debug(f"Prompting entity extraction LLM using {prompt}")

        response = self.client.chat(prompt=prompt)
        self.logger.info(f"Retrieved raw response from LLM: {response}")

        return json.loads(response)

    # def n_gram_splitter(self, phrase: str):
    #     spans = []
    #     split_phrase = phrase.split()
    #     for i in range(len(split_phrase)):
    #         for j in range(1, len(split_phrase) - i + 1):
    #             spans.append(" ".join(split_phrase[i:i+j]))
    #     return spans

    def fuzzy_search(self, phrases: list[str]):
        if not phrases:
            return []

        matches = []
        for phrase in phrases:
            matches += self.fuzzy_skb.query(phrase, return_scores=False)

        return matches

    def get_linked_context(self, question: str):
        extraction = self.extract(question)
        matches = self.fuzzy_search(extraction)

        prompt = f"""\n### Entity candidates

Here are some already fuzzy matched entities from the knowledge base, from some mentions in the question. You may pick which ones to use, as they are not always relevant. There may be a single match that is best.

{"\n".join(str(m) for m in matches)}"""

        return prompt
