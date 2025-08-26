import os
from dotenv import load_dotenv
import logging

import neo4j

from ..pkl.skb import SKB
from ..chroma_dbs.skb_chroma import Chroma_DB

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))

class Neo4j_DB:
    def __init__(self, collection_name: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.driver = neo4j.GraphDatabase.driver(uri=NEO4J_URI, auth=NEO4J_AUTH)
        self.database_name = collection_name

    def query(self, query: str, filter_ids: list[str] = None, other_params: dict[str, any] = None):
        with self.driver.session(database=self.database_name) as session:
            params = {}
            if filter_ids:
                params["ids"] = filter_ids
            if other_params:
                params = {**params, **other_params}  # Merge filter_ids params with other_params

            # Execute the query with the merged parameters
            result = session.run(query, **params)
            return [record.data() for record in result]

    def clear(self):
        with self.driver.session(database=self.database_name) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def template_insert_node(self, entity_label: str, props: dict[str, any]):
        prop_keys = ', '.join(f'{k}: ${k}' for k in props)
        return f"MERGE (n:{entity_label} {{{prop_keys}}})"

    def template_insert_relation(self, from_label, rel_name, to_label):
        return f"MATCH (a:{from_label} {{external_id: $from_id}}) MATCH (b:{to_label} {{external_id: $to_id}}) MERGE (a)-[r:{rel_name.upper()}]->(b)"

    def parse(self, skb: SKB, max_entities: int = None, clear_previous: bool = True):
        if clear_previous:
            self.clear()

        with self.driver.session(database=self.database_name) as session:
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

    def attach_chroma_embeddings(self, chromadb: Chroma_DB, max_rows: int = None):
        self.logger.info(f"Retrieving embeddings from Chroma collection {chromadb.collection_name}")
        if max_rows:
            entries = chromadb.collection.get(limit=max_rows, include=["embeddings"])
            self.logger.debug(f"Limited set of chroma entries: {entries}")
        else:
            entries = chromadb.collection.get(include=["embeddings"])

        batch_data = [
            {"id": id_val, "embedding": embedding}
            for id_val, embedding in zip(entries["ids"], entries["embeddings"])
        ]

        cypher_query = f"""
        UNWIND $batch as item
        MATCH (n {{external_id: item.id}})
        SET n.embedding = item.embedding
        """

        self.logger.info(f"Attaching embeddings from Chroma collection {chromadb.collection_name} to Neo4j database")
        with self.driver.session(database=self.database_name) as session:
            session.run(cypher_query, batch=batch_data)

        self.logger.info(f"Finished attaching embeddings from Chroma collection {chromadb.collection_name} to Neo4j database")

    def remove_embeddings(self):
        cypher_query = f"""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        REMOVE n.embedding
        """

        self.logger.info("Removing embeddings from Neo4j database")
        with self.driver.session(database=self.database_name) as session:
            session.run(cypher_query)

        self.logger.info("Finished removing embeddings from Neo4j database")

    def ftsearch(self, query: str):
        """Full-text search for fuzzy partial matching"""
        split_text = [f"{s}~" for s in query.split()]
        cypher_query = f"""
        CALL db.index.fulltext.queryNodes("names", "{" OR ".join(split_text)}")
        YIELD node, score
        WHERE score > 1
        RETURN apoc.text.join(LABELS(node), ", ") AS EntityType, COALESCE(node.name, node.description) AS TextValue, ROUND(score, 2) AS FullTextScore
        LIMIT 4
        """
        return self.query(cypher_query)