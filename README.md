# Honours - Heidi Leow

## Pipelines setup

### GraphRAG

Note that the prompts directory is the default on initialisation by the package. See https://microsoft.github.io/graphrag/get_started/ for more details

#### Indexing to KG

```
cd src/graphrag
graphrag index --root .
```

#### Querying

Local search for more specific questions:

```
cd src/graphrag
graphrag query \
--root . \
--method local \
--query "Which subcomponent of Power unit has an RPN of 27."
```

Global search for more high-level questions:

```
cd src/graphrag
graphrag query \
--root . \
--method global \
--query "What components have an RPN value over 40"
```

### Text-to-Cypher

#### Creating KG

Make sure that a Neo4j instance has separately been set up. Retrieve your username and the password for that instance, and replace it in the .env file.

#### Querying

```
python3 text_to_cypher.py "What is the average detection value over the full dataset?"
```
