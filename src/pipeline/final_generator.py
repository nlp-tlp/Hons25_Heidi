import logging

from .llm import ChatClient

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

final_generator_prompt = """You are the final generator in a RAG system. The user question that has to be answered is:

{question}

Answer this question using the following already retrieved context. Speak directly to the user who asked the question and if speaking of the retrieved records, pretend like you were the one that performed the retrieval directly. You may have to infer the names of certain terms (e.g. 'fm' may represent 'failure mode'). If no records are provided in the context, do not guess and simply say so. If you think some information is missing, just assume that the retrieved information is the correct set to answer the question, but mention this in your response:

{records}

Here is some context on the way the data was stored that might be of use:

{schema}"""

# Config - change as necessary
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)
logger = logging.getLogger("Generator")

# Generator
class FinalGenerator:
    def __init__(self, client: ChatClient):
        self.client = client

    def generate(self, question: str, retrieved_nodes: list[dict]):
        # Cases to use prewritten answers
        if len(retrieved_nodes) == 0:
            logger.info("No nodes retrieved, returning pre-written response.")
            return "No records could be found. Either the answer is that there are no such entities, or that the context given was insufficient to retrieve the right records. If you believe it is the latter, try rephrasing your question."

        if len(retrieved_nodes) > 50:
            logger.info(f"Too many nodes retrieved: {len(retrieved_nodes)}, returning pre-written response.")
            return "Too many records were retrieved. Either the answer contains that many entities, or the model gave a bad plan of retrieval. If you believe it is the latter, try entering the question again."

        # Build prompt
        context_string = "\n".join([str(r) for r in retrieved_nodes]) if retrieved_nodes else "No relevant records found."
        prompt = final_generator_prompt.format(
            question=question,
            records=context_string,
            schema=schema_context
        )
        logger.debug(f"Prompting LLM using: {prompt}")

        # Generate final response from LLM
        final_response = self.client.chat(prompt=prompt)
        return final_response