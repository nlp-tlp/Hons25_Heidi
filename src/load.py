import logging
import sys

# from databases import Neo4j_SKB, Te3s_SKB, Glove_SKB, Flair_SKB, Fuzzy_SKB, SKB, BarrickSchema, load_from_barrick_csv
from scopes import PropertyTextScopeGraph

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

# def load_skb():
#     skb = SKB(BarrickSchema)
#     load_from_barrick_csv(skb, "databases/pkl/fmea_barrick_filled.csv")
#     skb.save_pickle("databases/pkl/skb.pkl")

# def load_neo4j_skb():
#     skb_loaded = SKB(BarrickSchema)
#     skb_loaded.load_pickle("databases/pkl/skb.pkl")

#     neo4j_skb = Neo4j_SKB()
#     neo4j_skb.parse(skb_loaded)

# def load_te3s_skb():
#     skb_loaded = SKB(BarrickSchema)
#     skb_loaded.load_pickle("databases/pkl/skb.pkl")

#     te3s_skb = Te3s_SKB(collection_name="te3s_all")
#     te3s_skb.parse(skb_loaded, only_semantic=False)

# def load_glove_skb():
#     skb_loaded = SKB(BarrickSchema)
#     skb_loaded.load_pickle("databases/pkl/skb.pkl")

#     glove_skb = Glove_SKB()
#     glove_skb.parse(skb_loaded)

# def load_flair_skb():
#     skb_loaded = SKB(BarrickSchema)
#     skb_loaded.load_pickle("databases/pkl/skb.pkl")

#     flair_skb = Flair_SKB()
#     flair_skb.parse(skb_loaded)

# def load_fuzzy_skb():
#     skb_loaded = SKB(BarrickSchema)
#     skb_loaded.load_pickle("databases/pkl/skb.pkl")

#     fuzzy_skb = Fuzzy_SKB()
#     fuzzy_skb.parse(skb_loaded)
#     fuzzy_skb.save_pickle("databases/other/fuzzy_types.pkl")

# def attach_chroma_embeddings_to_neo4j():
#     te3s_skb = Te3s_SKB(collection_name="te3s_all")
#     neo4j_skb = Neo4j_SKB()

#     neo4j_skb.attach_chroma_embeddings(te3s_skb)

# def remove_embeddings_from_neo4j():
#     neo4j_skb = Neo4j_SKB()
#     neo4j_skb.remove_embeddings()

scope_graphs = {
    "property_text": PropertyTextScopeGraph
}

if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Incorrect number of arguments")
        exit(1)

    scope = sys.argv[1]
    if scope not in scope_graphs:
        print("Unrecognised scope graph")
        exit(1)
    scope_graph = scope_graphs[scope]()

    action = sys.argv[2]
    match action:
        case "skb":
            scope_graph.setup_skb(
                filepath="databases/pkl/fmea_barrick_filled.csv",
                outpath=f"databases/pkl/{scope}.pkl"
            )
        case "chroma":
            scope_graph.load_skb(skb_file=f"databases/pkl/{scope}.pkl")
            scope_graph.setup_chroma()
        case "neo4j":
            scope_graph.load_skb(skb_file=f"databases/pkl/{scope}.pkl")
            scope_graph.load_chroma()
            scope_graph.setup_neo4j()
        case default:
            print("Unrecognised action")
            exit(1)
