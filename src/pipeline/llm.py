import re

import logging
logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)
logger = logging.getLogger("LLM Client")

class ChatClient:
    """Consistent interface to access both local and cloud API-based models.
    All configurations for specific models should be set in a .env file to be loaded in.

    @param provider: LLM provider, currently one of ("ollama", "openai"). Uses "ollama" by default.
    @param model: Model provided by the provider, e.g. "llama3.2" for "ollama" or "gpt-4o" for "openai".
    @param temperature: The randomness of the model. Between 0.0 and 1.0.
    """

    def __init__(self, provider: str = "ollama", model: str = "llama3.2", temperature: float = 0.0):
        self.provider = provider
        self.model = model
        self.temperature = temperature

        if provider == "ollama":
            import ollama
            self.client = ollama.Client(
                host="http://localhost:11434",
            )
        elif provider == "openai":
            import openai
            import os
            from dotenv import load_dotenv
            load_dotenv()
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat(self, prompt: str) -> str:
        """Prompts the configured model.

        @param prompt: The prompt to send.
        """
        logger.info(f"Prompting Chat LLM at provider {self.provider} model {self.model}")
        if self.provider == "ollama":
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response['message']['content'].strip()

            if self.model == "deepseek-r1:14b":
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            return content
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()

class EmbeddingClient:
    """Consistent interface to access both local and cloud API-based embedding models. All configurations for specific models should be set in a .env file to be loaded in.

    @param provider: LLM provider, currently one of ("ollama", "openai"). Uses "ollama" by default.
    @param model: Model provided by the provider, e.g. "mxbai-embed-large" for "ollama" or "text-embedding-3-small" for "openai".
    @param dimensions: Number of dimensions for the produced vectors. Used if the model specified supports this setting.
    """

    def __init__(self, provider: str = "ollama", model: str = "mxbai-embed-large", dimensions: float | None = None):
        self.provider = provider
        self.model = model
        self.dimensions = dimensions

        if provider == "ollama":
            import ollama
            self.client = ollama.Client(
                host="http://localhost:11434",
            )
        elif provider == "openai":
            import openai
            import os
            from dotenv import load_dotenv
            load_dotenv()
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def embed(self, text: str):
        """Generates a list of embeddings from a given string, using the configured model.

        @param text: The text to embed.
        """
        logger.info(f"Prompting Embedding LLM at provider {self.provider} model {self.model}")
        if self.provider == "ollama":
            embedding = self.client.embed(model=self.model, input=text)
            return embedding["embedding"]
        elif self.provider == "openai":
            if self.dimensions:
                embedding = self.client.embeddings.create(
                    input=text,
                    model=self.model,
                    dimensions=self.dimensions
                )
            else:
                embedding = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
            return embedding.data[0].embedding
