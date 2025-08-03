import logging

from databases import Neo4j_SKB, Te3s_SKB, Glove_SKB, Flair_SKB, Fuzzy_SKB, SKB, BarrickSchema, load_from_barrick_csv

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

def load_skb():
    skb = SKB(BarrickSchema)
    load_from_barrick_csv(skb, "databases/pkl/fmea_barrick_filled.csv")
    skb.save_pickle("databases/pkl/skb.pkl")

def load_neo4j_skb():
    skb_loaded = SKB(BarrickSchema)
    skb_loaded.load_pickle("databases/pkl/skb.pkl")

    neo4j_skb = Neo4j_SKB()
    neo4j_skb.parse(skb_loaded)

def load_te3s_skb():
    skb_loaded = SKB(BarrickSchema)
    skb_loaded.load_pickle("databases/pkl/skb.pkl")

    te3s_skb = Te3s_SKB()
    te3s_skb.parse(skb_loaded)

def load_glove_skb():
    skb_loaded = SKB(BarrickSchema)
    skb_loaded.load_pickle("databases/pkl/skb.pkl")

    glove_skb = Glove_SKB()
    glove_skb.parse(skb_loaded)

def load_flair_skb():
    skb_loaded = SKB(BarrickSchema)
    skb_loaded.load_pickle("databases/pkl/skb.pkl")

    flair_skb = Flair_SKB()
    flair_skb.parse(skb_loaded)

def load_fuzzy_skb():
    skb_loaded = SKB(BarrickSchema)
    skb_loaded.load_pickle("databases/pkl/skb.pkl")

    fuzzy_skb = Fuzzy_SKB()
    fuzzy_skb.parse(skb_loaded)
    fuzzy_skb.save_pickle("databases/other/fuzzy_types.pkl")

def attach_chroma_embeddings_to_neo4j():
    te3s_skb = Te3s_SKB()
    neo4j_skb = Neo4j_SKB()

    neo4j_skb.attach_chroma_embeddings(te3s_skb)

