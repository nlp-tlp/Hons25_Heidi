from .chroma_dbs.skb_chroma import Chroma_DB, Te3sEmbeddingFunction, FlairEmbeddingFunction, GloveEmbeddingFunction
from .neo4j_dbs.skb_neo4j import Neo4j_DB
from .other.skb_fuzzy import Fuzzy_SKB

from .pkl.skb import SKB
from .pkl.skb_barrick import BarrickSchema, load_from_barrick_csv

# def embedder_factory(name: str):
#     embedders = {
#         'text-embedding-3-small': Chroma_DB(collection_name="te3s_all", embed_fnc=Te3sEmbeddingFunction),
#         'glove mean-pooling': Chroma_DB(collection_name="glove", embed_fnc=GloveEmbeddingFunction),
#         'flair mean-pooling': Chroma_DB(collection_name="flair", embed_fnc=FlairEmbeddingFunction)
#     }
#     return embedders[name]()