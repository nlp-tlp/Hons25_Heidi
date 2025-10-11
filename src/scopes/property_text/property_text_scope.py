import logging
import csv
import re
from pydantic import Field

from databases.pkl.skb import SKB, SKBSchema, SKBNode, SKBGraph
from databases import Chroma_DB, Neo4j_DB, Te3sEmbeddingFunction
from llm import ChatClient, EmbeddingClient
from linking import EntityLinker

class PropertyTextScopeSchema(SKBSchema):
    class Subsystem(SKBNode):
        name: str = Field(..., id=True)

    class Component(SKBNode):
        part_of: list[str] = Field(..., id=True, relation=True, dest='Subsystem')
        name: str = Field(..., id=True)

    class SubComponent(SKBNode):
        part_of: list[str] = Field(..., id=True, relation=True, dest='Component')
        name: str = Field(..., id=True)

    class FailureMode(SKBNode):
        for_part: list[str] = Field(..., id=True, relation=True, dest='SubComponent')
        related_to: list[str] = Field(..., relation=True, dest='FailureCause, FailureEffect')
        has_action: list[str] = Field(..., relation=True, dest='CurrentControls, RecommendedAction')
        description: str = Field(..., id=True, semantic=True)
        occurrence: int = Field(..., id=True)
        detection: int = Field(..., id=True)
        rpn: int = Field(..., id=True)
        severity: int = Field(..., id=True)

    class FailureEffect(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class FailureCause(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class RecommendedAction(SKBNode):
        description: str = Field(..., id=True, semantic=True)

    class CurrentControls(SKBNode):
        description: str = Field(..., id=True, semantic=True)

class PropertyTextScopeGraph(SKBGraph):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.schema = PropertyTextScopeSchema
        self.name = "property_text"
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

                subsystem = self.schema.Subsystem(name=row["Subsystem"].strip())
                subsystem_id = self.skb.add_entity(subsystem)

                component = self.schema.Component(part_of=[subsystem_id], name=row["Component"].strip())
                component_id = self.skb.add_entity(component)
                self.skb.get_entity_by_id(subsystem_id)._rev_in_subsystem = component_id

                subcomponent = self.schema.SubComponent(part_of=[component_id], name=row["Sub-Component"].strip())
                subcomponent_id = self.skb.add_entity(subcomponent)

                fe = self.schema.FailureEffect(description=row["Potential Effect(s) of Failure"].strip())
                fe_id = self.skb.add_entity(fe)

                fc = self.schema.FailureCause(description=row["Potential Cause(s) of Failure"].strip())
                fc_id = self.skb.add_entity(fc)

                actions = []
                controls_str = row["Current Controls"].strip()
                if controls_str:
                    controls = self.schema.CurrentControls(description=controls_str.strip())
                    controls_id = self.skb.add_entity(controls)
                    actions.append(controls_id)

                recommended_str = row["Recommended Action"].strip()
                if recommended_str:
                    recommended = self.schema.RecommendedAction(description=recommended_str.strip())
                    recommended_id = self.skb.add_entity(recommended)
                    actions.append(recommended_id)

                fm = self.schema.FailureMode(
                    for_part=[subcomponent_id],
                    related_to=[fe_id, fc_id],
                    has_action=actions,
                    description=row["Potential Failure Mode"].strip(),
                    occurrence=int(row["Occurrence"]),
                    detection=int(row["Detection"]),
                    rpn=int(row["RPN"]),
                    severity=int(row["Severity"])
                )
                self.skb.add_entity(fm)

        self.skb.save_pickle(outpath)

class PropertyTextScopeRetriever:
    def __init__(self,  graph: PropertyTextScopeGraph, prompt_path: str,
        allow_linking: bool, allow_extended: bool, allow_descriptive_only: bool,
        chat_client: ChatClient, embedding_client: EmbeddingClient = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.allow_linking = allow_linking # these options are only meaningful for this strat
        self.allow_extended = allow_extended
        self.allow_descriptive_only = allow_descriptive_only

        self.graph = graph
        self.chat_client = chat_client
        self.embedding_client = embedding_client
        self.linker = EntityLinker(client=ChatClient(provider="openai", model="gpt-4.1-mini-2025-04-14"), graph=self.graph)

        with open(prompt_path) as f:
            self.prompt = f.read()

    def retrieve(self, question: str):
        self.logger.info(f"Question given: {question}")

        # Entity linking
        linker_context = ""
        if self.allow_linking:
            linker_context = self.linker.get_linked_context(question)

        # Get LLM-generated Cypher
        query = self.generate_cypher(question, linker_context=linker_context)
        self.logger.info(f"Generated Cypher: {query}")

        # Process extended functions and run command
        return self.execute_query(query)

    def schema_context(self):
        tag_semantic = True if self.allow_descriptive_only else False
        return self.graph.schema.schema_to_jsonlike_str(tag_semantic=tag_semantic, tag_uniqueness=True)

    def generate_cypher(self, question: str, linker_context: str = ""):
        # Build prompt
        prompt = self.prompt.format(
            schema=self.schema_context(),
            question=question
        ) + linker_context
        self.logger.info(f"Prompting LLM using: {prompt}")

        # Generate Cypher from LLM
        raw_response = self.chat_client.chat(prompt=prompt)
        cypher_query = re.sub(r"^```[a-zA-Z]*\s*|```$", "", raw_response, flags=re.MULTILINE).strip() # Remove markdown if present
        return cypher_query

    def execute_query(self, query: str):
        original_query = query

        params = {}
        if self.allow_extended and self.embedding_client:
            query, params = self.convert_extended_functions(query)

        try:
            if params:
                records = self.graph.neo4j.query(query, other_params=params)
            else:
                records = self.graph.neo4j.query(query)
            self.logger.info(f"Retrieved {len(records)} records from Neo4j.")
            return original_query, records, None
        except Exception as e:
            self.logger.error(f"Error running Cypher: {e}")
            return original_query, [], f"Error during Cypher execution: {e}"

    def convert_extended_functions(self, query: str, semantic_threshold: float = 0.6418, fuzzy_threshold: float = 1.8):
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
