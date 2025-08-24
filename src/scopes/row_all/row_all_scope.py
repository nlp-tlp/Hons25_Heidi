import logging
import csv
import re
from pydantic import Field

from databases.pkl.skb import SKB, SKBSchema, SKBNode, SKBGraph
from databases import Chroma_DB, Neo4j_DB, Te3sEmbeddingFunction
from llm import ChatClient, EmbeddingClient

class RowAllScopeSchema(SKBSchema):
    class Row(SKBNode):
        contents: str = Field(..., id=True, semantic=True)

class RowAllScopeGraph(SKBGraph):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.schema = RowAllScopeSchema
        self.name = "row_all"
        self.embedding_func = Te3sEmbeddingFunction()

        self.skb: SKB = None
        self.chroma: Chroma_DB
        # self.neo4j: Neo4j_DB

    def setup_skb(self, filepath: str, outpath: str, max_rows: int = None):
        self.skb = SKB(self.schema)

        with open(filepath, 'r', encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break

                row_text = f"Subsystem: {row["Subsystem"].strip()} | Component: {row["Component"].strip()} | SubComponent: {row["Sub-Component"]} | FailureMode: {row["Potential Failure Mode"].strip()} | FailureEffect: {row["Potential Effect(s) of Failure"].strip()} | FailureCause: {row["Potential Cause(s) of Failure"].strip()} | CurrentControls: {row["Current Controls"].strip()} | RecommendedAction: {row["Recommended Action"].strip()} | Occurrence: {row["Occurrence"].strip()} | Detection: {row["Detection"].strip()} | Severity: {row["Severity"].strip()} | RPN: {row["RPN"].strip()}"
                row = self.schema.Row(
                    contents=row_text
                )
                self.skb.add_entity(row)

        self.skb.save_pickle(outpath)

class RowAllScopeRetriever:
    def __init__(self, graph: RowAllScopeGraph,
        chat_client: ChatClient, embedding_client: EmbeddingClient
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.graph = graph
        self.chat_client = chat_client
        self.embedding_client = embedding_client

    def retrieve(self, question: str, k=15, threshold=None):
        self.logger.info(f"Question given: {question}")

        # No Cypher generation here - just a vector search
        vector_matches = self.graph.chroma.query(
            query=question.strip(),
            k=k,
            threshold=threshold
        )
        return question, [{"content": match[2]} for match in vector_matches], None

    def schema_context(self):
        return self.graph.schema.schema_to_jsonlike_str(tag_semantic=False, tag_uniqueness=False)
