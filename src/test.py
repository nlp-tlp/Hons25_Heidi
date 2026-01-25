import logging
import sys
import inspect

from scopes import retriever_choices, retriever_factory

logging.basicConfig(
    level=logging.WARNING,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)
logging.getLogger().setLevel(logging.CRITICAL)

def test_prop_basic():
    applicable_retrievers = [1, 3]
    query = (
        f"MATCH (subcomponent:SubComponent)\n"
        f"RETURN COUNT(subcomponent.name) AS subcomponent_count"
    )
    return applicable_retrievers, query

def test_prop_basic_2():
    applicable_retrievers = [1, 3]
    query = (
        f"MATCH (subsystem:Subsystem)<-[:PART_OF]-(component:Component)<-[:PART_OF]-(subcomponent:SubComponent)<-[:FOR_PART]-(failuremode:FailureMode)\n"
        f"WHERE failuremode.rpn > 35\n"
        f"RETURN subsystem.name, component.name, COUNT(DISTINCT failuremode) AS failuremode_count, COLLECT(DISTINCT failuremode.description) AS failuremode_descriptions, COLLECT(DISTINCT subcomponent.name) AS subcomponents, COLLECT(DISTINCT failuremode.rpn) AS rpn_values"
    )
    return applicable_retrievers, query

def test_prop_sem():
    applicable_retrievers = [1, 3]
    query = (
        f"MATCH (subsystem:Subsystem)<-[:PART_OF]-(component:Component)<-[:PART_OF]-(subcomponent:SubComponent)<-[:FOR_PART]-(failuremode:FailureMode)\n"
        f"WHERE IS_SEMANTIC_MATCH(failuremode.description, 'Blocked')\n"
        f"RETURN subsystem.name, component.name, subcomponent.name, failuremode.description, failuremode.severity, failuremode.occurrence, failuremode.detection, failuremode.rpn"
    )
    return applicable_retrievers, query

def test_prop_fuzzy_sem():
    applicable_retrievers = [1]
    query = (
        f"MATCH (subsystem:Subsystem)<-[:PART_OF]-(component:Component)<-[:PART_OF]-(subcomponent:SubComponent)<-[:FOR_PART]-(failuremode:FailureMode)-[:HAS_ACTION]->(recommendedaction:RecommendedAction)\n"
        f"WHERE IS_FUZZY_MATCH(subsystem.name, 'power train')\n"
        f"\tAND IS_SEMANTIC_MATCH(recommendedaction.description, 'lubrication')\n"
        f"RETURN subsystem.name, component.name, subcomponent.name, failuremode.description, failuremode.occurrence, failuremode.detection, failuremode.severity, failuremode.rpn, recommendedaction.description"
    )
    return applicable_retrievers, query

def test_prop_multi_fuzzy_sem():
    applicable_retrievers = [1]
    query = (
        f"MATCH (subsystem:Subsystem)<-[:PART_OF]-(component:Component)<-[:PART_OF]-(subcomponent:SubComponent)<-[:FOR_PART]-(failuremode:FailureMode)-[:RELATED_TO]->(failurecause:FailureCause)\n"
        f"WHERE IS_FUZZY_MATCH(subsystem.name, 'hydraulic system')\n"
        f"\tAND IS_FUZZY_MATCH(component.name, 'cylinders')\n"
        f"\tAND IS_SEMANTIC_MATCH(failurecause.description, 'wearing')\n"
        f"RETURN subsystem.name, component.name, subcomponent.name, failuremode.description, failurecause.description, failuremode.occurrence, failuremode.detection, failuremode.severity, failuremode.rpn"
    )
    return applicable_retrievers, query

def test_prop_union_sem():
    applicable_retrievers = [1]
    query = (
        f"MATCH (system:SystemComponent)<-[:FOR_PART]-(failure:FailureOccurrence)\n"
        f"WHERE IS_FUZZY_MATCH(system.name, 'hydraulic system')\n"
        f"\tAND (\n"
        f"\tIS_SEMANTIC_MATCH(failure.description, 'blockage')\n"
        f"\tOR IS_SEMANTIC_MATCH(failure.description, 'valves fittings blockage')\n"
        f"\t)\n"
        f"RETURN system.name, failure.description, failure.severity, failure.occurrence, failure.detection, failure.rpn\n"
        f"UNION\n"
        f"MATCH (system:SystemComponent)<-[:FOR_PART]-(failure:FailureOccurrence)\n"
        f"WHERE IS_FUZZY_MATCH(system.name, 'valves fittings')\n"
        f"\tAND IS_SEMANTIC_MATCH(failure.description, 'blockage')\n"
        f"RETURN system.name, failure.description, failure.severity, failure.occurrence, failure.detection, failure.rpn\n"
    )
    return applicable_retrievers, query

def test_row_fuzzy_sem_with():
    applicable_retrievers = [6]
    query = (
        f"MATCH (row:Row)\n"
        f"WHERE IS_FUZZY_MATCH(row.contents, 'Power Unit')\n"
        f"\tAND IS_SEMANTIC_MATCH(row.contents, 'leak')\n"
        f"WITH row, row.occurrence AS occurrence, row.rpn AS rpn, row.severity AS severity, row.detection AS detection\n"
        f"ORDER BY occurrence DESC, rpn DESC, severity DESC, detection DESC\n"
        f"LIMIT 1\n"
        f"RETURN row.contents, occurrence, rpn, severity, detection"
    )
    return applicable_retrievers, query

if __name__ == "__main__":
    logger = logging.getLogger("QueryExecutionTesting")
    logger.setLevel(logging.INFO)

    retrievers = [retriever_factory(choice["name"], choice["allow_linking"]) for choice in retriever_choices[2:-1]]
    test_funcs = [obj for name, obj in inspect.getmembers(sys.modules[__name__])
        if (inspect.isfunction(obj) and name.startswith('test'))]

    logger.info("Retrievers loaded, running tests.")

    success_count = 0
    all_count = 0
    failures = []
    for test in test_funcs:
        applicable_retrievers, query = test()
        for ret in applicable_retrievers:
            _, _, error = retrievers[ret].execute_query(query)

            if (error):
                logger.info(f"{test.__name__} failed with error:\n\n{error}")
            else:
                success_count += 1
            all_count += 1

    logger.info(f"Tests finished. {success_count} / {all_count} passed.")
