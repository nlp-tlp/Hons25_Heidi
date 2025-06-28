import logging

from databases import Neo4j_SKB, Te3s_SKB, Glove_SKB, Flair_SKB, SKB, BarrickSchema

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

skb_loaded = SKB(BarrickSchema)
skb_loaded.load_pickle("databases/pkl/skb.pkl")

def load_neo4j_skb():
    neo4j_skb = Neo4j_SKB()
    neo4j_skb.parse(skb_loaded)

def load_te3s_skb():
    te3s_skb = Te3s_SKB()
    te3s_skb.parse(skb_loaded)

def load_glove_skb():
    glove_skb = Glove_SKB()
    glove_skb.parse(skb_loaded)

def load_flair_skb():
    flair_skb = Flair_SKB()
    flair_skb.parse(skb_loaded)
