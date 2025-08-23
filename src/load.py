import logging
import sys

# from databases import Neo4j_SKB, Te3s_SKB, Glove_SKB, Flair_SKB, Fuzzy_SKB, SKB, BarrickSchema, load_from_barrick_csv
from scopes import PropertyTextScopeGraph, RowTextScopeGraph

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

scope_graphs = {
    "property_text": PropertyTextScopeGraph,
    "row_text": RowTextScopeGraph
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
