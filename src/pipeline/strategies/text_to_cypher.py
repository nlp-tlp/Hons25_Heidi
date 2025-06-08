from neo4j import GraphDatabase
import openai

import logging
import re

from pipeline.final_generator import FinalGenerator

# Config - change as necessary
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)
logger = logging.getLogger("TextToCypher")

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
- FailureModes are unique per SubComponent. Failure modes with identical names on different systems with different causes and effects exist, and should be treated as separate.
- Similarly, SubComponents are unique per Component, and Components are unique per SubSystem.
- Integer properties on FailureMode require exact matching/range-based queries.
- All other fields are text-based, and require substring-matching/fuzzy-matching. This includes the names of the components and sub-components. Do not assume that the user has given the right spelling/ casing."""

text_to_cypher_prompt = """You are a system that converts natural language questions into Cypher queries.
Use the following schema to understand the Neo4j graph:

{schema}

The question you should convert is:

{question}

Only output the minimal Cypher query, with no markdown wrapping such that it can be directly executed as a query. Ensure that more than enough is retrieved - Include any information that might be useful to generate a response."""

final_generator_prompt = """You are the final generator in a RAG system. The user question that has to be answered is:

{question}

Answer this question using the following already retrieved context. Assume these are already the right answers to the question, and simply need to be put into natural language. If no records are provided in the context, do not guess and simply say so:

{records}"""

# Retriever
class TextToCypherRetriever:
    def __init__(self, model="gpt-4o"):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.model = model
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def retrieve(self, question: str | None):
        if question is None:
            logger.info("No question given, terminating")
            return
        logger.info(f"Question given: {question}")

        # Get generated Cypher
        query = self.generate_cypher(question)
        logger.info(f"Generated Cypher: {query}")

        # Run command
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = [record.data() for record in result]
            logger.info(f"Retrieved {len(records)} records from Neo4j.")
            return query, records
        except Exception as e:
            logger.error(f"Error running Cypher: {e}")
            return query, []

    def generate_cypher(self, question: str):
        # Build prompt
        prompt = text_to_cypher_prompt.format(
            schema=schema_context,
            question=question
        )
        logger.info(f"Prompting LLM using: {prompt}")

        # Generate Cypher from LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_content = response.choices[0].message.content.strip()
        cypher_query = re.sub(r"^```[a-zA-Z]*\s*|```$", "", raw_content, flags=re.MULTILINE).strip() # Remove markdown if present
        return cypher_query

# Example usage
# example_question = "What components have an RPN value over 35"
# retriever = TextToCypherRetriever(model="gpt-4o")
# retrieved_records = retriever.retrieve(question=example_question)

# generator = FinalGenerator(model="gpt-4o")
# final_response = generator.generate(question=example_question, retrieved_nodes=retrieved_records)

# logger.info(f"Final response: {final_response}")