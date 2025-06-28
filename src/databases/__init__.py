from .chroma_dbs.skb_chroma import Chroma_DB, Te3s_SKB, Glove_SKB, Flair_SKB
from .neo4j_dbs.skb_neo4j import Neo4j_SKB

from .pkl.skb import SKB
from .pkl.skb_barrick import BarrickSchema, load_from_barrick_csv

def embedder_factory(name: str):
    embedders = {
        'text-embedding-3-small': Te3s_SKB,
        'glove mean-pooling': Glove_SKB,
        'flair mean-pooling': Flair_SKB
    }
    return embedders[name]()