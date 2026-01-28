from pydantic import BaseModel, RootModel
import logging
import os
from dotenv import load_dotenv
import openai

chat_model_choices = [
    "gpt-4.1-2025-04-14", # main experiments in paper
    "gpt-5.2-2025-12-11"
]

class ChatClient:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, prompt: str, model: str = chat_model_choices[0], response_format: BaseModel | RootModel = None) -> str:
        if model is None:
            model = chat_model_choices[0]
        self.logger.info(f"Prompting Chat LLM at OpenAI model {model}")

        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.0001, # shouldn't matter, just in case it does
            seed=12345, # shouldn't matter
            presence_penalty = 0.0, # default
            frequency_penalty = 0.0, # default
            **({"response_format": response_format} if response_format is not None else {})
        )
        return response.choices[0].message.content.strip()

class EmbeddingClient:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, text: str, model: str = "text-embedding-3-small"):
        self.logger.info(f"Prompting Embedding LLM at OpenAI model {model}")
        embedding = self.client.embeddings.create(
            input=text,
            model=model
        )
        return embedding.data[0].embedding
