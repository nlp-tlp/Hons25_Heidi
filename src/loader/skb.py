from pydantic import BaseModel
import hashlib
import pickle
import json

from neo4j import GraphDatabase

class SKBSchema:
    @classmethod
    def schema_to_jsonlike(cls):
        schema_dict = {}
        for name, cls in vars(cls).items():
            if not (isinstance(cls, type) and issubclass(cls, SKBNode)):
                continue

            entity_dict = {}

            for field_name, field in cls.model_fields.items():
                meta = [field.annotation.__name__]

                if field.json_schema_extra.get("relation"):
                    meta.pop()
                    meta.append(f"relation_to {field.json_schema_extra.get('dest')}")
                if field.json_schema_extra.get("id"):
                    meta.append("@informs_uniqueness")
                if field.json_schema_extra.get("semantic"):
                    meta.append("@match_semantically")

                entity_dict[field_name] = ' '.join(meta)

            schema_dict[name] = entity_dict

        return schema_dict

    @classmethod
    def schema_to_jsonlike_str(cls):
        return json.dumps(cls.schema_to_jsonlike(), indent=4).replace('"', '')

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

class Neo4jSKB:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def query_insert_node(self, entity_label: str, props: dict[str, any]):
        prop_keys = ', '.join(f'{k}: ${k}' for k in props)
        return f"MERGE (n:{entity_label} {{{prop_keys}}})"

    def query_insert_relation(self, from_label, rel_name, to_label):
        return (
            f"MATCH (a:{from_label} {{external_id: $from_id}}), (b:{to_label} {{external_id: $to_id}}) "
            f"MERGE (a)-[r:{rel_name.upper()}]->(b)"
        )

    def parse(self, skb: SKB):
        self.clear()
        with self.driver.session() as session:
            # First pass: create entities
            for node_id, node in skb.get_entities().items():
                entity = node.__class__.__name__
                props = node.get_props()
                props['external_id'] = node_id
                query = self.query_insert_node(entity, props)
                session.run(query, props)

            # Second pass: create relations
            for node_id, node in skb.get_entities().items():
                from_label = node.__class__.__name__
                relations = node.get_relations()
                for rel_name, rel_targets in relations.items():
                    for target_id in rel_targets:
                        to_node = skb.get_entity_by_id(target_id)
                        to_label = to_node.__class__.__name__
                        query = self.query_insert_relation(from_label, rel_name, to_label)
                        session.run(query, {"from_id": node_id, "to_id": target_id})


