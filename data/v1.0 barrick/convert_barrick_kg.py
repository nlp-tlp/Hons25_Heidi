from neo4j import GraphDatabase
import csv
# import itertools

URI = "bolt://localhost:7687" # Default local URI, change as necessary
AUTH = ("neo4j", "fmea_barrick") # Change as necessary

driver = GraphDatabase.driver(URI, auth=AUTH)

def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")

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

def create_nodes(tx, row):
    query = """
    MERGE (sp:Spreadsheet {name: $spreadsheet})

    MERGE (s:Subsystem {name: $subsystem})
    MERGE (c:Component {name: $component})
    MERGE (sc:SubComponent {name: $sub_component})
    MERGE (s)-[:HAS_COMPONENT]->(c)
    MERGE (c)-[:HAS_SUB_COMPONENT]->(sc)

    MERGE (fm:FailureMode {name: $failure_mode, sub_component: $sub_component})
    MERGE (sc)-[:HAS_FAILURE_MODE]->(fm)

    MERGE (e:FailureEffect {name: $failure_effect})
    MERGE (fm)-[:HAS_EFFECT]->(e)

    MERGE (pc:FailureCause {name: $failure_cause})
    MERGE (fm)-[:CAUSED_BY]->(pc)

    // Conditionally create RecommendedAction node
    FOREACH (_ IN CASE WHEN $recommended_action IS NOT NULL THEN [1] ELSE [] END |
        MERGE (a:RecommendedAction {name: $recommended_action})
        MERGE (fm)-[:HAS_RECOMMENDED_ACTION]->(a)
    )

    // Conditionally create CurrentControls node
    FOREACH (_ IN CASE WHEN $current_controls IS NOT NULL THEN [1] ELSE [] END |
        MERGE (ctrl:CurrentControls {name: $current_controls})
        MERGE (fm)-[:HAS_CONTROLS]->(ctrl)
    )

    SET fm.occurrence = toInteger($occurrence),
        fm.detection = toInteger($detection),
        fm.rpn = toInteger($rpn),
        fm.severity = toInteger($severity)

    MERGE (fm)-[:IN_SPREADSHEET]->(sp)
    """
    tx.run(query, **row)

def process_csv(filepath):
    with driver.session() as session, open(filepath, 'r', encoding='utf-8') as file:
        session.execute_write(clear_database)

        reader = csv.DictReader(file)

        # for row in itertools.islice(reader, 4): # Use this if you just want top rows for inspection
        for row in reader:
            mapped_row = {v: row[k].strip() if row[k] else None for k, v in column_map.items() if k in row}
            session.execute_write(create_nodes, mapped_row)


process_csv("fmea_barrick_filled.csv")
driver.close()

