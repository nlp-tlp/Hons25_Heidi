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
    def __init__(self, graph: RowTextScopeGraph, prompt_path: str,
        allow_descriptive_only: bool,
        chat_client: ChatClient, embedding_client: EmbeddingClient
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.allow_descriptive_only = allow_descriptive_only

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
        tag_semantic = True if self.allow_descriptive_only else False
        return self.graph.schema.schema_to_jsonlike_str(tag_semantic=tag_semantic, tag_uniqueness=True)

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

    def convert_extended_functions(self, query: str, semantic_threshold: float = 0.6586, fuzzy_threshold: float = 0.42):
        query = self.escape_parens_in_strings(query)

        # Fuzzy match replacement
        fuzzy_matches = []
        if self.allow_descriptive_only:
            fuzzy_matches = list(re.finditer(r"IS_FUZZY_MATCH\(([^,]+),\s*([^)]+)\)", query))
            fuzzy_subqueries = []
            fuzzy_var_names = []
            for i, match in enumerate(fuzzy_matches, 1):
                target = match.group(1).strip()
                target_entity = target.split('.')[0]
                search_phrase = match.group(2)[1:-1].replace("__LPAREN__", "").replace("__RPAREN__", "")

                split_text = [f"{s}~" for s in search_phrase.split()]
                fuzzy_list_var = f"fuzzy_list_{i}"
                fuzzy_score_var = f"fuzzy_score_{i}"

                # Subquery to collect matching nodes
                subquery = (
                    f"\nCALL () {{\n"
                    f"  CALL db.index.fulltext.queryNodes('names', '{' OR '.join(split_text)}')\n"
                    f"  YIELD node AS node_{i}, score AS {fuzzy_score_var}\n"
                    f"  WHERE {fuzzy_score_var} > {fuzzy_threshold}\n"
                    f"  RETURN collect(node_{i}) AS {fuzzy_list_var}\n"
                    f"}}"
                )
                fuzzy_subqueries.append(subquery)
                fuzzy_var_names.append((target_entity, fuzzy_list_var))

            def fuzzy_in_replacer(match):
                target, fuzzy_list_var = fuzzy_var_names.pop(0)
                return f"{target} IN {fuzzy_list_var}"

        # Semantic match replacement
        where_matches = list(re.finditer(
            r"(WHERE\s+)(.*?)(?=\s+(RETURN|WITH|ORDER BY|SKIP|LIMIT|MATCH|UNWIND|CALL|CREATE|MERGE|SET|DELETE|REMOVE|FOREACH|LOAD CSV|OPTIONAL MATCH|$))",
            query,
            re.IGNORECASE | re.DOTALL
        ))
        if not where_matches:
            query = self.unescape_parens_in_strings(query)
            self.logger.info(f"Converted query to: {query}")
            return query, None

        i = 1
        params = {}
        for where_match in where_matches:
            semantic_matches = re.findall(r"IS_SEMANTIC_MATCH\(([^,]+),\s*([^)]+)\)", where_match.group(0))

            with_clause = "WITH *"
            new_where_clause = where_match.group(0)
            for target, search_phrase in semantic_matches:
                search_phrase_clean = search_phrase[1:-1] # take off apostrophes
                self.logger.info(f"Processing semantic match for: {search_phrase_clean}")
                vector = self.embedding_client.embed(search_phrase_clean.strip().lower())

                vector_placeholder = f"vector_{i}"
                similarity_var = f"similarity_{i}"
                target_entity = target.split('.')[0]
                i += 1

                with_clause += f", vector.similarity.cosine({target_entity}.embedding, ${vector_placeholder}) AS {similarity_var}"
                new_where_clause = re.sub(rf"IS_SEMANTIC_MATCH\(\s*{target}\s*,\s*{search_phrase}\s*\)", f"{similarity_var} > {semantic_threshold}", new_where_clause)
                params[vector_placeholder] = vector

            if semantic_matches:
                query = query.replace(where_match.group(0), f"{with_clause} {new_where_clause}")

        # Needs to be here to avoid triggering the semantic matching regex
        if fuzzy_matches:
            # Split query on UNION (preserving UNIONs)
            union_parts = re.split(r'(\s+UNION\s+)', query, flags=re.IGNORECASE)
            if len(union_parts) > 1:
                rebuilt_query = ""
                fuzzy_pattern = r"IS_FUZZY_MATCH\(([^,]+),\s*([^)]+)\)"

                fuzzy_var_names_copy = fuzzy_var_names.copy()  # To avoid mutation issues

                for idx in range(0, len(union_parts), 2):
                    branch = union_parts[idx]
                    union = union_parts[idx+1] if idx+1 < len(union_parts) else ""

                    branch_fuzzy_matches = list(re.finditer(fuzzy_pattern, branch))
                    branch_fuzzy_subqueries = []
                    branch_fuzzy_var_names = []

                    for i, match in enumerate(branch_fuzzy_matches, 1):
                        target_entity, fuzzy_list_var = fuzzy_var_names_copy.pop(0)
                        branch_fuzzy_var_names.append((target_entity, fuzzy_list_var))
                        branch_fuzzy_subqueries.append(fuzzy_subqueries.pop(0))

                    def branch_fuzzy_in_replacer(match):
                        target, fuzzy_list_var = branch_fuzzy_var_names.pop(0)
                        return f"{target} IN {fuzzy_list_var}"

                    # Replace fuzzy matches in branch
                    if branch_fuzzy_matches:
                        branch = re.sub(fuzzy_pattern, branch_fuzzy_in_replacer, branch)
                        branch = "\n".join(branch_fuzzy_subqueries) + branch

                    rebuilt_query += branch + union
                query = rebuilt_query
            else:
                query = re.sub(
                    r"IS_FUZZY_MATCH\(([^,]+),\s*([^)]+)\)",
                    fuzzy_in_replacer,
                    query
                )
                query = "\n".join(fuzzy_subqueries) + "\n" + query

        query = self.unescape_parens_in_strings(query)
        self.logger.info(f"Converted query to: {query}")
        return query, params

    def escape_parens_in_strings(self, text: str):
        string_pattern = re.compile(r"(['\"])(.*?)(\1)", re.DOTALL)

        def replacer(match):
            quote, content, end = match.groups()
            escaped_content = content.replace("(", "__LPAREN__").replace(")", "__RPAREN__")
            return f"{quote}{escaped_content}{end}"

        return string_pattern.sub(replacer, text)

    def unescape_parens_in_strings(self, text: str):
        return text.replace("__LPAREN__", "(").replace("__RPAREN__", ")")