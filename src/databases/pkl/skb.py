from pydantic import BaseModel
import hashlib
import pickle
import json

class SKBSchema:
    @classmethod
    def schema_to_jsonlike(cls, tag_semantic: bool = True, tag_uniqueness: bool = True):
        schema_dict = {}
        for name, cls in vars(cls).items():
            if not (isinstance(cls, type) and issubclass(cls, SKBNode)):
                continue

            entity_dict = {}

            for field_name, field in cls.model_fields.items():
                meta = [field.annotation.__name__]

                if field.json_schema_extra.get("relation"):
                    field_name = field_name.upper()
                    meta.pop()
                    meta.append(f"@relation_to {field.json_schema_extra.get('dest')}")
                if tag_uniqueness and field.json_schema_extra.get("id"):
                    meta.append("@informs_uniqueness")
                if tag_semantic and field.json_schema_extra.get("semantic"):
                    meta.append("@match_semantically")

                entity_dict[field_name] = ' '.join(meta)

            schema_dict[name] = entity_dict

        return schema_dict

    @classmethod
    def schema_to_jsonlike_str(cls, tag_uniqueness: bool = True, tag_semantic: bool = True):
        return json.dumps(cls.schema_to_jsonlike(tag_uniqueness, tag_semantic), indent=4).replace('"', '')

class SKBNode(BaseModel):
    def get_props(self) -> dict[str, any]:
        return { k: v for k, v in self.model_dump().items()
            if not self.model_fields[k].json_schema_extra.get("relation", False)}

    def get_relations(self) -> dict[str, any]:
        return { k: v for k, v in self.model_dump().items()
            if self.model_fields[k].json_schema_extra.get("relation", False)}

    def get_identity(self) -> dict[str, any]:
        return { k: v for k, v in self.model_dump().items()
            if self.model_fields[k].json_schema_extra.get("id", False)}

    def get_semantic(self) -> dict[str, any]:
        return { k: v for k, v in self.model_dump().items()
            if self.model_fields[k].json_schema_extra.get("semantic", False)}

    def get_textual(self) -> dict[str, any]:
        return { k: v for k, v in self.model_dump().items()
            if isinstance(v, str)}

    def compute_id(self) -> str:
        id_vals = self.get_identity().values()
        return hashlib.sha1("|".join(str(val) for val in id_vals).encode()).hexdigest()

class SKB:
    def __init__(self, schema: SKBSchema):
        self.schema = schema
        self.nodes: dict[str, dict[str, any]] = {}

    def add_entity(self, entity: SKBNode) -> str:
        node_id = entity.compute_id()
        if node_id not in self.nodes:
            self.nodes[node_id] = entity
        else: # Merge non-identity fields
            existing = self.nodes[node_id]
            for k, v in entity.model_dump().items():
                if existing.model_fields[k].json_schema_extra.get("id", False):
                    continue
                if isinstance(v, list): # Only adding for list items for now
                    existing_list = getattr(existing, k)
                    merged = list(set(existing_list + v)) # Add only new unique items
                    setattr(existing, k, merged)
        return node_id

    def get_entities(self):
        return self.nodes

    def get_entity_by_id(self, id: str):
        return self.nodes[id]

    def save_pickle(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.nodes, f)

    def load_pickle(self, path: str):
        with open(path, "rb") as f:
            self.nodes = pickle.load(f)

class SKBGraph:
    def load_skb(self, skb_file: str):
        self.skb = SKB(self.schema)
        self.skb.load_pickle(skb_file)

    def setup_chroma(self):
        from databases import Chroma_DB
        self.chroma = Chroma_DB(collection_name=self.name, embed_fnc=self.embedding_func)
        self.chroma.parse(self.skb)

    def load_chroma(self):
        from databases import Chroma_DB
        self.chroma = Chroma_DB(collection_name=self.name, embed_fnc=self.embedding_func)

    def setup_neo4j(self):
        from databases import Neo4j_DB
        self.neo4j = Neo4j_DB(collection_name=self.name.replace("_", "-"))
        self.neo4j.parse(self.skb)
        self.neo4j.attach_chroma_embeddings(self.chroma)

    def load_neo4j(self):
        from databases import Neo4j_DB
        self.neo4j = Neo4j_DB(collection_name=self.name.replace("_", "-"))
