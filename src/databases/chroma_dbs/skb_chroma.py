import os
from dotenv import load_dotenv
import logging
import shutil
import numpy as np

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
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = PersistentClient(path=CHROMA_DB_PATH)

        self.collection: Collection = None
        self.collection_name: str = None
        self.embed_fnc = None

    def clear(self):
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

    def load(self):
        if not self.collection_name or not self.embed_fnc:
            return

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fnc,
            metadata={"hnsw:space": "cosine"}
        )

    def query(self, query: str, k: int = 25, threshold: float = None, filter_entity: str = None, filter_ids: list[str] = None):
        params = {}
        if filter_entity:
            params["where"] = {"type": filter_entity}
        if filter_ids:
            params["ids"] = filter_ids

        query_result: QueryResult = self.collection.query(
            query_texts=[query],
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

    def parse(self, skb: SKB, max_nodes: int = None, clear_previous: bool = True):
        if clear_previous:
            self.clear()

        docs = []
        docs_meta = []
        docs_ids = []
        for i, (node_id, node) in enumerate(skb.get_entities().items()):
            if max_nodes is not None and i >= max_nodes:
                break

            semantic_fields = node.get_semantic()
            if not semantic_fields:
                continue

            text = " | ".join(v.lower() for v in semantic_fields.values())
            meta = {"type": type(node).__name__}

            docs.append(text)
            docs_meta.append(meta)
            docs_ids.append(node_id)

        self.collection.upsert(
            documents=docs,
            metadatas=docs_meta,
            ids=docs_ids
        )

class Te3s_SKB(Chroma_DB):
    def __init__(self):
        super().__init__()

        self.embed_fnc = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        self.collection_name = "te3s"
        self.load()

class Glove_SKB(Chroma_DB):
    def __init__(self):
        super().__init__()

        self.embed_fnc = Glove_SKB.GloveEmbeddingFunction()
        self.collection_name = "glove"
        self.load()

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

class Flair_SKB(Chroma_DB):
    def __init__(self):
        super().__init__()

        self.embed_fnc = Flair_SKB.FlairEmbeddingFunction()
        self.collection_name = "flair"
        self.load()

    class FlairEmbeddingFunction(EmbeddingFunction[Documents]):
        def __init__(self):
            stacked_flair_embedding = StackedEmbeddings([
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
