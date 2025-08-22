import logging
import csv
from pydantic import Field

from databases.pkl.skb import SKB, SKBSchema, SKBNode, SKBGraph
from databases import Chroma_DB, Neo4j_DB

class RowTextScopeSchema(SKBSchema):
    class Row(SKBNode):
        contents: str = Field(..., id=True)
        occurrence: int = Field(..., id=True)
        detection: int = Field(..., id=True)
        rpn: int = Field(..., id=True)
        severity: int = Field(..., id=True)

class RowTextScopeGraph(SKBGraph):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.schema = RowTextScopeSchema
        self.name = "property_text"

        self.skb: SKB = None
        self.chroma: Chroma_DB
        self.neo4j: Neo4j_DB

    def setup_skb(self, filepath: str, max_rows: int = None):
        pass

