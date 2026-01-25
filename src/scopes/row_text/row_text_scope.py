import logging
import csv
import re
from pydantic import Field

from databases.pkl.skb import SKB, SKBSchema, SKBNode, SKBGraph
from databases import Chroma_DB, Neo4j_DB, Te3sEmbeddingFunction
from llm import ChatClient, EmbeddingClient

class RowTextScopeSchema(SKBSchema):
    class Row(SKBNode):
        contents: str = Field(..., id=True, semantic=True, concats_fields="FailureMode, FailureEffect, FailureCause, Subsystem, Component, SubComponent, CurrentControls, RecommendedAction")
        occurrence: int = Field(..., id=True)
        detection: int = Field(..., id=True)
        rpn: int = Field(..., id=True)
        severity: int = Field(..., id=True)

class RowTextScopeGraph(SKBGraph):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.schema = RowTextScopeSchema
        self.name = "row_text"
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

                row_text = f"Subsystem: {row["Subsystem"].strip()} | Component: {row["Component"].strip()} | SubComponent: {row["Sub-Component"]} | FailureMode: {row["Potential Failure Mode"].strip()} | FailureEffect: {row["Potential Effect(s) of Failure"].strip()} | FailureCause: {row["Potential Cause(s) of Failure"].strip()} | CurrentControls: {row["Current Controls"].strip()} | RecommendedAction: {row["Recommended Action"].strip()}"
                row = self.schema.Row(
                    contents=row_text,
                    occurrence=int(row["Occurrence"]),
                    detection=int(row["Detection"]),
                    rpn=int(row["RPN"]),
                    severity=int(row["Severity"])
                )
                self.skb.add_entity(row)

        self.skb.save_pickle(outpath)

class RowTextScopeRetriever:
    def __init__(self, prompt_path: str, allow_descriptive_only: bool):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.allow_descriptive_only = allow_descriptive_only

        self.graph = RowTextScopeGraph()
        self.graph.load_neo4j()
        self.chat_client = ChatClient()
        self.embedding_client = EmbeddingClient()

        with open(prompt_path) as f:
            self.prompt = f.read()

    def retrieve(self, question: str, model: str = None):
        self.logger.info(f"Question given: {question}")

        # Get LLM-generated Cypher
        query = self.generate_cypher(question, model=model)
        self.logger.info(f"Generated Cypher: {query}")

        # Process extended functions and run command
        return self.execute_query(query)

    def schema_context(self):
        tag_semantic = True if self.allow_descriptive_only else False
        return self.graph.schema.schema_to_jsonlike_str(tag_semantic=tag_semantic, tag_uniqueness=True)

    def generate_cypher(self, question: str, model: str = None):
        # Build prompt
        prompt = self.prompt.format(
            schema=self.schema_context(),
            question=question
        )
        self.logger.info(f"Prompting LLM using: {prompt}")

        # Generate Cypher from LLM
        raw_response = self.chat_client.chat(prompt=prompt, model=model)
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

    def convert_extended_functions(self, query: str, semantic_threshold: float = 0.6586, fuzzy_threshold: float = 0.42):
        query = self.escape_parens_in_strings(query)

        # Semantic match replacement
        where_matches = list(re.finditer(r"(WHERE\s+)(.*?)(?=\s+(CALL|CREATE|DELETE|DETACH|EXISTS|FOREACH|LOAD|MATCH|MERGE|OPTIONAL|REMOVE|RETURN|SET|START|UNION|UNWIND|WITH|LIMIT|ORDER|SKIP|WHERE|YIELD|$))", query, re.IGNORECASE | re.DOTALL))
        if not where_matches:
            query = self.unescape_parens_in_strings(query)
            self.logger.info(f"Converted query to: {query}")
            return query, None

        params = {}
        for where_match in where_matches:
            semantic_matches = re.findall(r"IS_SEMANTIC_MATCH\(([^,]+),\s*([^)]+)\)", where_match.group(0))
            if not semantic_matches:
                continue

            new_with_clause = "WITH *"
            new_where_clause = where_match.group(0)
            for semantic_match_num, (target, search_phrase) in enumerate(semantic_matches):
                self.logger.info(f"Processing embedding for: {search_phrase[1:-1].strip().lower()}")
                vector = self.embedding_client.embed(search_phrase[1:-1].strip().lower()) # take off apostrophes and normalise
                vector_placeholder = f"vector_{semantic_match_num}"
                similarity_var = f"similarity_{semantic_match_num}"

                new_with_clause += f", vector.similarity.cosine({target.split('.')[0]}.embedding, ${vector_placeholder}) AS {similarity_var}"
                new_where_clause = re.sub(rf"IS_SEMANTIC_MATCH\(\s*{target}\s*,\s*{search_phrase}\s*\)", f"{similarity_var} > {semantic_threshold}", new_where_clause)
                params[vector_placeholder] = vector
            query = query.replace(where_match.group(), f"{new_with_clause}\n{new_where_clause}")

        # Fuzzy match replacement
        if not self.allow_descriptive_only:
            query = self.unescape_parens_in_strings(query)
            self.logger.info(f"Converted query to: {query}")
            return query, params

        rebuilt_query = ""
        union_branches = re.split(r'\s+UNION\s+', query, flags=re.IGNORECASE)
        for branch in union_branches:
            fuzzy_matches = re.findall(r"IS_FUZZY_MATCH\(([^,]+),\s*([^)]+)\)", branch)
            if not fuzzy_matches:
                continue

            new_branch = branch
            fuzzy_subqueries = []
            fuzzy_list_vars = []
            for fuzzy_match_num, (target, search_phrase) in enumerate(fuzzy_matches):
                split_text = [f"{s}~" for s in search_phrase[1:-1].replace("__LPAREN__", "").replace("__RPAREN__", "").replace("-", " ").split()]
                fuzzy_list_var = f"fuzzy_list_{fuzzy_match_num}"
                fuzzy_score_var = f"fuzzy_score_{fuzzy_match_num}"

                subquery = (
                    f"\nCALL () {{\n"
                    f"  CALL db.index.fulltext.queryNodes('names', '{' OR '.join(split_text)}')\n"
                    f"  YIELD node AS node_{fuzzy_match_num}, score AS {fuzzy_score_var}\n"
                    f"  WHERE {fuzzy_score_var} > {fuzzy_threshold}\n"
                    f"  RETURN collect(node_{fuzzy_match_num}) AS {fuzzy_list_var}\n"
                    f"}}\n"
                )
                fuzzy_subqueries.append(subquery)
                new_branch = re.sub(rf"IS_FUZZY_MATCH\(\s*{target}\s*,\s*{search_phrase}\s*\)", f"{target.split('.')[0]} IN {fuzzy_list_var}", new_branch)
                fuzzy_list_vars.append(fuzzy_list_var)

            with_matches = list(re.finditer(r"(WITH\s+)(.*?)(?=\s+(CALL|CREATE|DELETE|DETACH|EXISTS|FOREACH|LOAD|MATCH|MERGE|OPTIONAL|REMOVE|RETURN|SET|START|UNION|UNWIND|WITH|LIMIT|ORDER|SKIP|WHERE|YIELD|$))", new_branch, re.IGNORECASE | re.DOTALL))
            for with_match in with_matches:
                if "*" not in with_match.group():
                    new_branch = new_branch.replace(with_match.group(), f"{with_match.group()}, {", ".join(fuzzy_list_vars)}")

            if rebuilt_query:
                rebuilt_query += "\nUNION\n"
            rebuilt_query += "\n".join(fuzzy_subqueries) + new_branch

        if rebuilt_query:
            query = rebuilt_query
        query = self.unescape_parens_in_strings(query)
        self.logger.info(f"Converted query to: {query}")
        return query, params

    def escape_parens_in_strings(self, text: str):
        def replacer(match):
            quote_start, quoted_content, quote_end = match.groups()
            escaped_content = quoted_content.replace("(", "__LPAREN__").replace(")", "__RPAREN__")
            return f"{quote_start}{escaped_content}{quote_end}"

        return re.sub(r"(['\"])(.*?)(\1)", replacer, text, re.DOTALL)

    def unescape_parens_in_strings(self, text: str):
        return text.replace("__LPAREN__", "(").replace("__RPAREN__", ")")
