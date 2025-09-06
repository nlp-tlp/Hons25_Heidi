import os
from dotenv import load_dotenv
import logging
import shutil
import re

from chromadb import Collection, QueryResult, Documents, PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, EmbeddingFunction
import sqlite3
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence

from ..pkl.skb import SKB

load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Chroma_DB:
    def __init__(self, collection_name: str, embed_fnc):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.collection_name = collection_name
        self.embed_fnc = embed_fnc

        self.client = PersistentClient(path=CHROMA_DB_PATH)
        self.collection: Collection = None
        self.load()

    def load(self):
        """Load existing database collection."""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fnc,
            metadata={"hnsw:space": "cosine"}
        )

        self.logger.info(f"Collection {self.collection_name} loaded with size {self.collection.count()}")

    def clear(self):
        """Remove database collection."""
        if not self.collection:
            return

        self.logger.info(f"Deleting existing collection for {self.collection_name}")
        self.client.delete_collection(self.collection_name)

        # Remove metadata directories (couldn't find built-in functionality)
        conn = sqlite3.connect(os.path.join(CHROMA_DB_PATH, f"{os.path.basename(CHROMA_DB_PATH)}.sqlite3"))
        cursor = conn.cursor()
        cursor.execute("select s.id from segments s where s.scope='VECTOR';")
        current_collections = [row[0] for row in cursor]

        subfolders = [f.path for f in os.scandir(CHROMA_DB_PATH) if f.is_dir()]
        for subfolder in subfolders:
            if os.path.basename(subfolder) not in current_collections:
                self.logger.info(f"Removing metadata subfolder: {subfolder}")
                shutil.rmtree(subfolder)

        conn.execute("VACUUM")
        conn.close()

        # Re-initialise the collection
        self.load()

    def query(self, query: str, k: int = 25, threshold: float = None, filter_entities: list[str] = None, filter_ids: list[str] = None):
        """Vector embedding search."""
        params = {}
        if filter_entities:
            params["where"] = {"type": {"$in": filter_entities}}
        if filter_ids:
            params["ids"] = filter_ids

        query_result: QueryResult = self.collection.query(
            query_texts=[query.strip().lower()],
            n_results=k,
            **params
        )

        results = []
        for i in range(len(query_result["ids"][0])):
            # Thresholding is implemented like this because ChromaDB does not provide this functionality
            similarity = 1 - query_result["distances"][0][i]
            if threshold and similarity < threshold:
                break

            results.append([
                query_result["ids"][0][i],
                query_result["metadatas"][0][i]["type"],
                query_result["documents"][0][i],
                similarity
            ])

        return results

    def parse(self, skb: SKB, max_nodes: int = None, clear_previous: bool = True, only_semantic: bool = False):
        """Parse SKB content into Chroma database collection."""
        if clear_previous:
            self.clear()

        docs = []
        docs_meta = []
        docs_ids = []
        for i, (node_id, node) in enumerate(skb.get_entities().items()):
            if max_nodes is not None and i >= max_nodes:
                break

            semantic_fields = node.get_semantic() if only_semantic else node.get_textual()
            if not semantic_fields:
                continue

            text = " | ".join(self.preprocess_string(v) for v in semantic_fields.values())
            meta = {"type": type(node).__name__}

            docs.append(text)
            docs_meta.append(meta)
            docs_ids.append(node_id)

        self.collection.add(
            documents=docs,
            metadatas=docs_meta,
            ids=docs_ids
        )

        self.logger.info(f"New collection size: {self.collection.count()}")

    def preprocess_string(self, text: str):
        if not text:
            return ""

        text = text.lower()
        text = text.rstrip(".,") # Remove end with comma or dot point
        text = re.sub(r'\s+', ' ', text) # Remove double whitespaces
        return text

class Te3sEmbeddingFunction(OpenAIEmbeddingFunction):
    def __init__(self):
        super().__init__(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )

class GloveEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        glove_embedding = WordEmbeddings("glove")
        self.model = DocumentPoolEmbeddings([glove_embedding], pooling="mean")

    def __call__(self, input: Documents) -> list[list[float]]:
        if not input:
            return []

        embeddings = []
        for doc in input:
            sentence = Sentence(doc)
            self.model.embed(sentence)
            embedding = sentence.embedding.detach().numpy()
            embeddings.append(embedding)
        return embeddings

class FlairEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        stacked_flair_embedding = StackedEmbeddings([
            WordEmbeddings("glove"),
            FlairEmbeddings("news-forward"),
            FlairEmbeddings("news-backward")
        ])
        self.model = DocumentPoolEmbeddings([stacked_flair_embedding], pooling="mean")

    def __call__(self, input: Documents) -> list[list[float]]:
        if not input:
            return []

        embeddings = []
        for doc in input:
            sentence = Sentence(doc)
            self.model.embed(sentence)
            embedding = sentence.embedding.detach().numpy()
            embeddings.append(embedding)
        return embeddings