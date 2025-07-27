import re

def convert_extended_functions(query: str):
    # Fuzzy match replacement
    query = re.sub(r"IS_FUZZY_MATCH\(([^,]+),\s*([^)]+)\)", r"apoc.text.fuzzyMatch(\1, \2)", query)

    # Semantic match replacement
    match_params = re.findall(r"IS_SEMANTIC_MATCH\(([^,]+),\s*([^)]+)\)", query)
    # search_phrase = match_params[0]
    # vector = self.embedding_client.embed(search_phrase)
    # params = {"vector": vector}
    if not match_params:
        return query, None

    params = {}
    for i, (search_phrase, target) in enumerate(match_params):
        # vector = self.embedding_client.embed(search_phrase.strip())

        vector_placeholder = f"vector_{i}"
        query = query.replace(
            f"IS_SEMANTIC_MATCH({search_phrase}, {target})",
            f"vector.similarity.cosine(${vector_placeholder}, {target}) > 0.33"
        )
        params[vector_placeholder] = [0.1, 0.2, 0.3]

    return query, params

query = """MATCH (c:Component)<-[:PART_OF]-(sc:SubComponent)<-[:FOR_PART]-(fm:FailureMode)-[:HAS_ACTION]->(ra:RecommendedAction)
WHERE IS_FUZZY_MATCH(c.name, 'cabin controls') AND IS_FUZZY_MATCH(sc.name, 'wiper motor')
    AND IS_FUZZY_MATCH(fm.description, 'corrosion')
RETURN c.name, sc.name, fm.description, ra.description
"""

if __name__ == "__main__":
    print(convert_extended_functions(query)[0])