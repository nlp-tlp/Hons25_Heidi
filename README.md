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
--query "What is the RPN for condenser"
```

Global search for more high-level questions:

```
cd src/graphrag
graphrag query \
--root . \
--method global \
--query "What components have an RPN value over 40"
```
