from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import RagTemplate, GraphRAG

import logging
import argparse

# load in env variables
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))

logging
class KeywordFilter(logging.Filter):
    def __init__(self, keywords):
        super().__init__()
        self.keywords = keywords

    def filter(self, record):
        return any(keyword in record.getMessage() for keyword in self.keywords)

logger = logging.getLogger()
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()],
)

# Remove default handlers to prevent duplicate logs
for handler in logger.handlers:
    logger.removeHandler(handler)

handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

keywords_to_include = ["Text2CypherRetriever", "Context:"]
handler.addFilter(KeywordFilter(keywords_to_include))
logger.addHandler(handler)

# setup
neo4j_schema = """
Entities:
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
- All other fields are text-based, and require substring-matching/fuzzy-matching. This includes the names of the components and sub-components. Do not assume that the user has given the right spelling/ casing.

Other:
- Ensure that more than enough is retrieved. Include any information that might be useful to generate a response.
"""

prompt_template = RagTemplate(
    system_instructions="Answer the user question using the already-retrieved context provided context. Assume these are already the right answers to the question, and simply need to be put into natural language. If no Records are provided in the context, do not guess and simply say so.",
    template=""""Context:
{context}

Question:
{query_text}
""",
    expected_inputs=["context", "query_text"]
)

driver = GraphDatabase.driver(uri=NEO4J_URI, auth=NEO4J_AUTH)
llm = OpenAILLM(model_name="gpt-4o", api_key=OPENAI_API_KEY)
retriever = Text2CypherRetriever(driver=driver, llm=llm, neo4j_schema=neo4j_schema)
rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=prompt_template)

# interface for retrieval and generation
def run_query(question):
    retrieved_data = rag.search(query_text=question)
    generated_answer = retrieved_data.answer
    print("\n*** RESPONSE: ***\n", generated_answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question to the FMEA Knowledge Graph.")
    parser.add_argument("question", type=str, help="Enter a natural language question.")
    args = parser.parse_args()

    run_query(args.question)
