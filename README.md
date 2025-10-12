# Honours 2025 - Heidi Leow

This repository contains the code and QA dataset associated with the dissertation paper: "Keep Facts as Facts: Hybrid Graph-Based RAG for Semi-Structured FMEA Spreadsheets".

## Dataset

The locations for the dataset and QA set are located under the `data` directory:

1. The `dataset` subdirectory includes the file `fmea_dataset_filled.csv`, which is the main data used for loading and experimentation.
2. The `questions` subdirectory includes the file `fmea_qa_model.xlsx`, which includes the evaluation questions, model answers, and associated operation type categories and nuggets.

## Loading

To run the code in this repository, follow the instructions in this section to set up packages and configurations for external systems.

### Requirements

To install required packages for Python, run inside your virtual environment:

```shell
pip install -r requirements.txt
```

### Configuration for External Systems

A configuration file with the name `.env` should be set up with your details. A template version is located under `.env template`. Duplicate and rename this file, and fill in the values.

An instance of **Neo4j** should be set up and running on your system, as a new project. Change the details for the correct URI, and the username and password set to access this project.

An **OpenAI** key is also needed to access GPT models. Replace the placeholder with your key value.

### Loading Data

Once systems have been set up, the `load.py` file is set up as a pseudo-command-line interface. Calling it with the appropriate arguments will set up related structures according to the paper.

First, run:

```shell
cd src
python3 load.py property_text skb
```

Then, to fully set up a structure in (`property_text`, `concept_text`, `row_text`, `row_all`), run:

```shell
python3 load.py [structure] chroma
python3 load.py [structure] neo4j # except row_all
```

The relevant graph structure will be set up and ready to query in other code/ the Streamlit interface set up below.

### Evaluation

Structures that have been set up are accessible to evaluation code. The `evaluate.py` file is set up in a similar way to `load.py`.

To automatically extract nuggets from as set of model answers (already done in model answer Excel files):

```shell
cd src
python3 evaluate.py property_descriptive nugget
```

Then, to run do a full experiment run-through for a strategy defined in `src/scopes/__init__.py` use:

```shell
python3 evaluate.py [strategy] rag
```

The RAG responses will be located under the `evaluation/experiment_runs` directory. To use LLM-as-a-Judge run:

```shell
python3 evaluate.py [strategy] eval
```

Alternatively, the same metric calculation after manual marking can be done with:

```shell
python3 evaluate.py [strategy] metric
```

The evaluations are appended to the same files in the directory.

## Interface

The work of this project involves a Streamlit interface that allows easy access to the configured RAG strategies and vector search collections.

To use the interface, run the following commands:

```shell
cd src
python3 -m streamlit run app/streamlit_app.py
```

This will automatically open your browser to the app's local address. Alternatively, to run/refresh the interface without starting a new browser tab use:

```shell
cd src
python3 -m streamlit run app/streamlit_app.py --server.headless true
```
