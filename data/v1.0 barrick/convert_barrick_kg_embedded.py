from neo4j import GraphDatabase
import openai

import csv
import time

# Config, change as necessary
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

column_map = {
    "Spreadsheet": "spreadsheet",
    "Subsystem": "subsystem",
    "Component": "component",
    "Sub-Component": "sub_component",
    "Potential Failure Mode": "failure_mode",
    "Potential Effect(s) of Failure": "failure_effect",
    "Severity": "severity",
    "Potential Cause(s) of Failure": "failure_cause",
    "Occurrence": "occurrence",
    "Current Controls": "current_controls",
    "Detection": "detection",
    "RPN": "rpn",
    "Recommended Action": "recommended_action"
}

# Database management
class GraphBuilderFMEA:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def get_embedding(self, text: str):
        if not text or not str(text).strip():
            return None

        # TODO: Retry logic not tested yet
        retries = 3
        for attempt in range(retries):
            try:
                response = openai.embeddings.create(
                    input=[text], model=EMBEDDING_MODEL, dimensions=256
                )
                return response.data[0].embedding
            except openai.RateLimitError:
                print("Rate limit hit. Retrying...")
                time.sleep(5)  # or exponential backoff
            except Exception as e:
                print(f"Error getting embedding: {e}")
                return None
        return None

    def create_row_nodes(self, tx, row: dict[str, any]):
        failure_mode_emb = self.get_embedding(row["failure_mode"])
        failure_cause_emb = self.get_embedding(row["failure_cause"])
        failure_effect_emb = self.get_embedding(row["failure_effect"])
        recommended_action_emb = self.get_embedding(row["recommended_action"])
        current_controls_emb = self.get_embedding(row["current_controls"])
        # NOTE: It is also possible to embed the system and components names - for simplicity not done for now

        query = """
        MERGE (sp:Spreadsheet {name: $spreadsheet})

        MERGE (s:Subsystem {name: $subsystem})
        MERGE (c:Component {name: $component})
        MERGE (sc:SubComponent {name: $sub_component})
        MERGE (s)-[:HAS_COMPONENT]->(c)
        MERGE (c)-[:HAS_SUB_COMPONENT]->(sc)

        MERGE (fm:FailureMode {name: $failure_mode, sub_component: $sub_component})
        SET fm.embedding = $failure_mode_emb,
            fm.occurrence = toInteger($occurrence),
            fm.detection = toInteger($detection),
            fm.rpn = toInteger($rpn),
            fm.severity = toInteger($severity)
        MERGE (sc)-[:HAS_FAILURE_MODE]->(fm)

        MERGE (e:FailureEffect {name: $failure_effect})
        SET e.embedding = $failure_effect_emb
        MERGE (fm)-[:HAS_EFFECT]->(e)

        MERGE (pc:FailureCause {name: $failure_cause})
        SET pc.embedding = $failure_cause_emb
        MERGE (fm)-[:CAUSED_BY]->(pc)

        // Conditionally create RecommendedAction node
        FOREACH (_ IN CASE WHEN $recommended_action IS NOT NULL THEN [1] ELSE [] END |
            MERGE (a:RecommendedAction {name: $recommended_action})
            SET a.embedding = $recommended_action_emb
            MERGE (fm)-[:HAS_RECOMMENDED_ACTION]->(a)
        )

        // Conditionally create CurrentControls node
        FOREACH (_ IN CASE WHEN $current_controls IS NOT NULL THEN [1] ELSE [] END |
            MERGE (ctrl:CurrentControls {name: $current_controls})
            SET ctrl.embedding = $current_controls_emb
            MERGE (fm)-[:HAS_CONTROLS]->(ctrl)
        )

        MERGE (fm)-[:IN_SPREADSHEET]->(sp)
        """
        tx.run(query, **row,
            failure_mode_emb=failure_mode_emb,
            failure_cause_emb=failure_cause_emb,
            failure_effect_emb=failure_effect_emb,
            recommended_action_emb=recommended_action_emb,
            current_controls_emb=current_controls_emb)


    def load_from_csv(self, filepath: str, max_rows: int = None):
        self.clear()
        with self.driver.session() as session, open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                mapped_row = {
                    v: row[k].strip() if row.get(k) else None
                    for k, v in column_map.items()
                }
                session.execute_write(self.create_row_nodes, mapped_row)

# Usage
builder = GraphBuilderFMEA(NEO4J_URI, NEO4J_AUTH)
builder.load_from_csv("fmea_barrick_filled.csv", max_rows=None)
builder.close()
