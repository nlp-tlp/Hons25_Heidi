import logging
import csv
import re
from pydantic import Field

from databases.pkl.skb import SKB, SKBSchema, SKBNode, SKBGraph
from databases import Chroma_DB, Neo4j_DB, Te3sEmbeddingFunction
from llm import ChatClient, EmbeddingClient

class ConceptTextScopeSchema(SKBSchema):
    class SystemComponent(SKBNode):
        name: str = Field(..., id=True, semantic=True)

    class FailureOccurrence(SKBNode):
        for_part: list[str] = Field(..., id=True, relation=True, dest="SystemComponent")
        related_to: list[str] = Field(..., relation=True, dest="ControlAction")
        description: str = Field(..., id=True, semantic=True)

    class ControlAction(SKBNode):
        description: str = Field(..., id=True, semantic=True)

class ConceptTextScopeGraph(SKBGraph):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.schema = ConceptTextScopeSchema
        self.name = "concept_text"
        self.embedding_func = Te3sEmbeddingFunction()

        self.skb: SKB = None
        self.chroma: Chroma_DB
        self.neo4j: Neo4j_DB

    def setup_skb(self, filepath: str, outpath: str, max_rows: int = None):
        self.skb = SKB(self.schema)

        with open(filepath, 'r', encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break

                system_text = f"Subsystem: {row["Subsystem"].strip()} | Component: {row["Component"].strip()} | SubComponent: {row["Sub-Component"]}"
                system = self.schema.SystemComponent(name=system_text)
                system_id = self.skb.add_entity(system)


                controls_str = row["Current Controls"].strip()
                recommended_str = row["Recommended Action"].strip()
                control_text = ""
                if controls_str:
                    control_text += f"CurrentControls: {controls_str}"
                if recommended_str:
                    control_text += " | " if controls_str else ""
                    control_text += f"RecommendedAction: {recommended_str}"

                actions = []
                if controls_str or recommended_str:
                    control = self.schema.ControlAction(description=control_text)
                    control_id = self.skb.add_entity(control)
                    actions.append(control_id)

                failure_text = f"FailureMode: {row["Potential Failure Mode"].strip()} | FailureEffect: {row["Potential Effect(s) of Failure"].strip()} | FailureCause: {row["Potential Cause(s) of Failure"].strip()}"
                failure = self.schema.FailureOccurrence(
                    for_part=[system_id],
                    related_to=actions,
                    description=failure_text
                )
                self.skb.add_entity(failure)

        self.skb.save_pickle(outpath)

class ConceptTextScopeRetriever:
    def __init__(self, graph: ConceptTextScopeGraph, prompt_path: str,
        chat_client: ChatClient, embedding_client: EmbeddingClient
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.graph = graph
        self.chat_client = chat_client
        self.embedding_client = embedding_client

        with open(prompt_path) as f:
            self.prompt = f.read()

    def retrieve(self, question: str):
        self.logger.info(f"Question given: {question}")

        # Get LLM-generated Cypher
        query = self.generate_cypher(question)
        self.logger.info(f"Generated Cypher: {query}")

        # Process extended functions and run command
        return self.execute_query(query)

    def schema_context(self):
        return self.graph.schema.schema_to_jsonlike_str(tag_semantic=False, tag_uniqueness=False)

    def generate_cypher(self, question: str):
        # Build prompt
        prompt = self.prompt.format(
            schema=self.schema_context(),
            question=question
        )
        self.logger.info(f"Prompting LLM using: {prompt}")

        # Generate Cypher from LLM
        raw_response = self.chat_client.chat(prompt=prompt)
        cypher_query = re.sub(r"^```[a-zA-Z]*\s*|```$", "", raw_response, flags=re.MULTILINE).strip() # Remove markdown if present
        return cypher_query

    def execute_query(self, query: str):
        original_query = query

        query, params = self.convert_extended_functions(query)

        try:
            records = self.graph.neo4j.query(query, other_params=params)

            self.logger.info(f"Retrieved {len(records)} records from Neo4j.")
            return original_query, records, None
        except Exception as e:
            self.logger.error(f"Error running Cypher: {e}")
            return original_query, [], f"Error during Cypher execution: {e}"

    def convert_extended_functions(self, query: str, semantic_threshold: float = 0.665):
        # Semantic match replacement
        where_matches = list(re.finditer(
            r"(WHERE\s+)(.*?)(?=\s+(RETURN|WITH|ORDER BY|SKIP|LIMIT|MATCH|UNWIND|CALL|CREATE|MERGE|SET|DELETE|REMOVE|FOREACH|LOAD CSV|OPTIONAL MATCH|$))",
            query,
            re.IGNORECASE | re.DOTALL
        ))
        if not where_matches:
            return query, None

        i = 1
        params = {}
        for where_match in where_matches:
            semantic_matches = re.findall(r"IS_SEMANTIC_MATCH\(([^,]+),\s*([^)]+)\)", where_match.group(0))

            with_clause = "WITH *"
            new_where_clause = where_match.group(0)
            for target, search_phrase in semantic_matches:
                self.logger.info(f"Processing semantic match for: {search_phrase}")
                vector = self.embedding_client.embed(search_phrase.strip())

                vector_placeholder = f"vector_{i}"
                similarity_var = f"similarity_{i}"
                target_entity = target.split('.')[0]
                i += 1

                with_clause += f", vector.similarity.cosine({target_entity}.embedding, ${vector_placeholder}) AS {similarity_var}"
                new_where_clause = re.sub(rf"IS_SEMANTIC_MATCH\(\s*{target}\s*,\s*{search_phrase}\s*\)", f"{similarity_var} > {semantic_threshold}", new_where_clause)
                params[vector_placeholder] = vector

            query = query.replace(where_match.group(0), f"{with_clause} {new_where_clause}")

        self.logger.info(f"Converted query to: {query}")
        return query, params
