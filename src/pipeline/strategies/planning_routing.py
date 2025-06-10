from neo4j import GraphDatabase

import logging
import uuid
import json
import re

from ..llm import ChatClient, EmbeddingClient

# Config - change as necessary
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

logging.basicConfig(
    level=logging.DEBUG,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)
logger = logging.getLogger("Agentic")

# Prompts
schema_context = """Entities:
- (Spreadsheet {name: STRING})
- (Subsystem {name: STRING})
- (Component {name: STRING})
- (SubComponent {name: STRING})
- (FailureMode {name: STRING, occurrence: INT, detection: INT, rpn: INT, severity: INT})
- (FailureEffect {name: STRING})
- (FailureCause {name: STRING})
- (RecommendedAction {name: STRING})  [OPTIONAL]
- (CurrentControls {name: STRING})  [OPTIONAL]

Relationships:
- (Spreadsheet)-[:CONTAINS]->(Subsystem)
- (Subsystem)-[:HAS_COMPONENT]->(Component)
- (Component)-[:HAS_SUB_COMPONENT]->(SubComponent)
- (SubComponent)-[:HAS_FAILURE_MODE]->(FailureMode)
- (FailureMode)-[:HAS_EFFECT]->(FailureEffect)
- (FailureMode)-[:CAUSED_BY]->(FailureCause)
- (FailureMode)-[:HAS_RECOMMENDED_ACTION]->(RecommendedAction)  [IF EXISTS]
- (FailureMode)-[:HAS_CONTROLS]->(CurrentControls)  [IF EXISTS]
- (FailureMode)-[:IN_SPREADSHEET]->(Spreadsheet)

Constraints:
- FailureModes are unique per Subsystem, Component, SubComponent combinations. Failure modes with identical names on different systems with different causes and effects exist, and should be treated as separate. When answering questions, the hierarchy should be specified unless obvious.
- Integer properties on FailureMode require exact matching/range-based queries.
- All other fields are text-based, and require substring-matching/fuzzy-matching. Do not assume that the user has given the right spelling/ casing in the question, and do not assume the data already in the system is correctly spelled either.

Embeddings: Attached on the nodes FailureMode, FailureEffect, FailureCause, RecommendedAction, CurrentControls. Only embeds the textual information in that individual node."""

planning_prompt = """You are a system that converts natural language questions into retrieval plans. There are two possible modes of retrieval - vector embedding search and KG Cypher queries. Use the following schema and information to understand the Neo4j graph:

{schema}

The question you should convert is:

{question}

Decompose this question into components that work best with structured KG search and unstructured semantic search. Output in only the following format, without any description, explanation, or formatting:

[
  {{"type": "kg", "query": "<Cypher query string>"}},
  {{"type": "vector", "search": "<semantic search string>"}},
  ...
]

Only decompose if needed - often a single step is sufficient. For each KG query component that is not the final one, only return the node id. Do not use '$' variables in the queries.

If not the first retrieval, the nodes from the previous result is stored in the variable "n" (already as full nodes, so do not add a filter clause for id filtering). Assume that subsequent steps will be given the same node type for "n" as the ones in the last "kg" component's return (e.g. if the last return were FailureEffect nodes, then the next vector search and KG queries will have n as the relevant nodes of the FailureEffect type. If you use something like `MATCH (n)-[:HAS_EFFECT]->(...)`, it will fail — because FailureEffect nodes don't have outgoing HAS_EFFECT relationships, and the direction may need to be reversed). Ensure that subsequent Cypher queries take into account this input type and do not use the wrong properties for the wrong type.

Vector searches will only return the node id, and in the same node type it was given. Properties need to be reretrieved via Cypher if necessary. The final step should return all relevant properties (instead of full nodes, e.g. RETURN fm.name, fm.rpn, ... instead of fm) needed to address the original question. More properties can be returned than necessary if they would make for a more informative natural language answer, however the nodes these properties are retrieved from must be defined in the same query - Assume that variables are not persisted across queries."""

# Individual strategies
class PlannerRetriever:
    def __init__(self, client: ChatClient, embedding_client: EmbeddingClient):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.client = client
        self.embedding_client = embedding_client

        self.session_id = str(uuid.uuid4())
        self.working_node_ids = []

    def retrieve(self, question: str | None):
        if question is None:
            logger.info("No question given, terminating")
            return
        logger.info(f"Question given: {question}")

        # Clear session
        self.clear_session()

        # Generate plan
        plan_text = self.generate_plan(question)

        try:
            plan = json.loads(plan_text)
        except Exception as e:
            logger.error(f"Error parsing plan: {e}")
            return plan_text, [], f"Error during plan parsing: {e}"

        # Loop through plan, execute KG or vector query
        try:
            for i, step in enumerate(plan):
                is_final = i == len(plan) - 1
                restrict = i > 0  # restrict to previous results

                if step["type"] == "kg":
                    records = self.kg_query(step["query"], restrict=restrict, final=is_final)
                    logger.info(f"Cypher query finished - Open pool of {len(records)} records from Neo4j.")

                    if len(records) == 0:
                        return plan, [], None
                elif step["type"] == "vector":
                    records = self.vector_search(step["search"], restrict=restrict, final=is_final)
                    logger.info(f"Vector search finished - Open pool of {len(records)} records from Neo4j.")

                    if len(records) == 0:
                        return plan, [], None
                else:
                    raise ValueError(f"Unknown plan type: {step['type']}")

                if is_final:
                    return plan, records, None
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return plan, [], f"Error during plan execution: {e}"

    def generate_plan(self, question: str):
        # Build prompt
        prompt = planning_prompt.format(
            schema=schema_context,
            question=question
        )
        logger.debug(f"Prompting LLM using: {prompt}")

        # Generate plan and component queries from LLM
        raw_response = self.client.chat(prompt=prompt)
        plan_text = re.sub(r"^```[a-zA-Z]*\s*|```$", "", raw_response, flags=re.MULTILINE).strip() # Remove markdown if present

        logger.info(f"Generated Plan:\n{plan_text}")
        return plan_text

    def kg_query(self, query: str, restrict=False, final=False):
        with self.driver.session() as session:
            if restrict and self.working_node_ids:
                full_query = f"""
                UNWIND $ids AS id
                MATCH (n) WHERE elementId(n) = id
                WITH DISTINCT n
                {query}
                """
                params = {"ids": self.working_node_ids}
            else:
                full_query = query
                params = {}

            logging.info(f"Running Cypher query:\n{full_query}")
            result = session.run(full_query, **params)
            records = [record.data() for record in result]

            # Clean up and store intermediate results
            if not final:
                # print("RECORDS", self.remove_embeddings(records))
                self.working_node_ids = self.extract_element_ids(records) # Update working set only if not final
            else:
                records = self.remove_embeddings(records) # Remove embedding value to reduce size of final prompt

            return records

    def vector_search(self, search: str, threshold=0.65, restrict=False, final=False):
        search_embedding = self.embedding_client.embed(text=search)

        with self.driver.session() as session:
            filter_clause = ""
            params = {
                "embedding": search_embedding,
                "threshold": threshold
            }

            if restrict and self.working_node_ids:
                filter_clause = """
                UNWIND $ids AS id
                MATCH (n) WHERE elementId(n) = id
                AND n.embedding IS NOT NULL
                """
                params["ids"] = self.working_node_ids
            else:
                filter_clause = """
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                """

            query = f"""
            {filter_clause}
            WITH n, vector.similarity.cosine(n.embedding, $embedding) AS score
            WHERE score > $threshold
            RETURN n, score, elementId(n) AS id
            """

            logging.info(f"Running vector search:\n{query}")
            result = session.run(query, **params)
            records = [record.data() for record in result]

            # Clean up and store intermediate results
            if not final:
                # print("RECORDS", self.remove_embeddings(records))
                self.working_node_ids = self.extract_element_ids(records) # Update working set only if not final
            else:
                records = self.remove_embeddings(records) # Remove embedding value to reduce size of final prompt

            return records

    def clear_session(self):
        with self.driver.session() as session:
            query = """MATCH (n)
            WHERE n.retrieval_session = $session_id
            REMOVE n.retrieval_session
            """

            session.run(query, session_id=self.session_id)
        logging.info("Cleared last Neo4j session")

    def extract_element_ids(self, records):
        ids = []
        for record in records:
            id = record.get("id")  # works for both KG/vector
            if id:
                ids.append(id)
        return ids

    def remove_embeddings(self, records):
        for record in records:
            for key, value in record.items():
                if isinstance(value, dict) and "embedding" in value:
                    del value["embedding"]
        return records


# Example usage
# example_question = "In the Fuel System component, what are the failure modes that are associated with temperature issues as a failure effect (temperature issues in failure effect node not failure mode)? Let me know what the associated failure effects were as well along with the failure modes."
# retriever = PlannerRetriever(model="gpt-4o")
# retrieved_records = retriever.retrieve(question=example_question)

# generator = FinalGenerator(model="gpt-4o")
# final_response = generator.generate(question=example_question, retrieved_nodes=retrieved_records)

# logger.info(f"Final response: {final_response}")
