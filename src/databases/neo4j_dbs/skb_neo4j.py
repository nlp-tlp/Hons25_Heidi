import os
from dotenv import load_dotenv
import logging

import neo4j

from ..pkl.skb import SKB

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))

class Neo4j_DB:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.driver = neo4j.GraphDatabase.driver(uri=NEO4J_URI, auth=NEO4J_AUTH)

    def query(self, query: str, filter_ids: list[str] = None):
        with self.driver.session() as session:
            if filter_ids:
                result = session.run(query, **{"ids": filter_ids})
            else:
                result = session.run(query)
            return [record.data() for record in result]

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def template_insert_node(self, entity_label: str, props: dict[str, any]):
        prop_keys = ', '.join(f'{k}: ${k}' for k in props)
        return f"MERGE (n:{entity_label} {{{prop_keys}}})"

    def template_insert_relation(self, from_label, rel_name, to_label):
        return (
            f"MATCH (a:{from_label} {{external_id: $from_id}}), (b:{to_label} {{external_id: $to_id}}) "
            f"MERGE (a)-[r:{rel_name.upper()}]->(b)"
        )

class Neo4j_SKB(Neo4j_DB):
    def parse(self, skb: SKB, max_entities: int = None, clear_previous: bool = True):
        if clear_previous:
            self.clear()

        with self.driver.session() as session:
            # First pass: create entities
            for i, (node_id, node) in enumerate(skb.get_entities().items()):
                if max_entities is not None and i >= max_entities:
                    break

                entity = node.__class__.__name__
                props = node.get_props()
                props['external_id'] = node_id
                query = self.template_insert_node(entity, props)
                session.run(query, props)

            # Second pass: create relations
            for i, (node_id, node) in enumerate(skb.get_entities().items()):
                if max_entities is not None and i >= max_entities:
                    break

                from_label = node.__class__.__name__
                relations = node.get_relations()
                for rel_name, rel_targets in relations.items():
                    for target_id in rel_targets:
                        to_node = skb.get_entity_by_id(target_id)
                        to_label = to_node.__class__.__name__
                        query = self.template_insert_relation(from_label, rel_name, to_label)
                        session.run(query, {"from_id": node_id, "to_id": target_id})

