import logging
import json
from typing import Callable
import csv

from llm import ChatClient

# Prompts
PROMPT_PATH = "evaluation/output_mapping_prompt.txt"

# Evaluator
class PlanOutputEvaluator:
    def __init__(self, mapper_client: ChatClient):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.mapper_client = mapper_client
        with open(PROMPT_PATH, 'r') as f:
            self.output_mapping_prompt = f.read()

    def generate(self, plan_generation_function: Callable[[str], str], model_answers_filepath: str, generated_answers_filepath: str, start_id: int = 1):
        with open(model_answers_filepath) as f:
            model_queries = json.load(f)

        # Generate queries/ plans
        generated_queries = []
        for entry in model_queries:
            generated_query = plan_generation_function(question=entry["question"])
            generated_queries.append({"id": entry["id"], "generated_query": generated_query})

        # Initialise JSON file for writing
        with open(generated_answers_filepath, 'w') as gen_answers_file:
            json_generated_queries = json.dumps(generated_queries)
            gen_answers_file.write(json_generated_queries)

    def evaluate(self, plan_execution_function: Callable[[str], list[dict[str, any]]], model_answers_filepath: str, generated_answers_filepath: str, metrics_filepath: str, start_id: int = 1):
        with open(model_answers_filepath) as f:
            gold_queries = json.load(f)

        with open(generated_answers_filepath) as f:
            generated_queries = json.load(f)

        # Execute until end or error
        metrics_store = [{"question_id": "avg", "precision_all": 0, "recall_all": 0, "precision_col": 0, "recall_col": 0, "query_length": 0, "order_correctness": 0}]
        for gold_query, generated_query in zip(gold_queries, generated_queries):
            entry_id = gold_query["id"]
            entry_eval_config = gold_query["eval_config"]

            # Execute
            try:
                gold_output = plan_execution_function(gold_query["ground_truth_query"])
                generated_output = plan_execution_function(generated_query["generated_query"])
            except Exception as e:
                self.logger.error(f"Error during query/ plan execution on ID {entry_id}: {e}")
                break

            # Deal with no output
            if len(generated_output) == 0 or len(gold_output) == 0:
                metrics = {
                    "question_id": entry_id,
                    "query_length": len(generated_query["generated_query"]) / len(gold_query["ground_truth_query"]),
                }

                if len(generated_output) == 0 and len(gold_output) == 0:
                    metrics.update({"precision_all": 1.0, "recall_all": 1.0, "precision_col": 1.0, "recall_col": 1.0, "order_correctness": 1.0})
                elif len(generated_output) != 0:
                    metrics.update({"precision_all": 0.0, "recall_all": 1.0, "precision_col": 0.0, "recall_col": 1.0, "order_correctness": 0.0})
                else:
                    metrics.update({"precision_all": 1.0, "recall_all": 0.0, "precision_col": 1.0, "recall_col": 0.0, "order_correctness": 1.0})
                metrics_store.append(metrics)
                for key in ["precision_all", "recall_all", "precision_col", "recall_col", "query_length", "order_correctness"]:
                    metrics_store[0][key] += metrics[key]
                break

            # Normalise generated output if aliases are different
            gold_field_names = list(gold_output[0].keys())
            generated_field_names = list(generated_output[0].keys())

            if not set(gold_field_names).issubset(set(generated_field_names)):
                field_mappings = self.predict_alias_mappings(gold_aliases=gold_field_names, candidate_aliases=generated_field_names)
                print(f"MAPPING: {field_mappings}")

                generated_output = [{field_mappings.get(k, k): v for k, v in entry.items()} for entry in generated_output]

            # Calculate and score evaluation
            metrics = {"question_id": entry_id}
            metrics.update(self.calc_row_metrics(gold_output=gold_output, generated_output=generated_output, eval_config=entry_eval_config))
            metrics.update(self.calc_col_metrics(gold_aliases=gold_field_names, normalised_candidate_aliases=list(generated_output[0].keys()), optional_columns=entry_eval_config["optional_columns"]))
            metrics["query_length"] = len(generated_query["generated_query"]) / len(gold_query["ground_truth_query"])
            metrics_store.append(metrics)

            for key in ["precision_all", "recall_all", "precision_col", "recall_col", "query_length", "order_correctness"]:
                metrics_store[0][key] += metrics[key]

        for key in ["precision_all", "recall_all", "precision_col", "recall_col", "query_length", "order_correctness"]:
            metrics_store[0][key] = round(metrics_store[0][key] / (len(metrics_store) - 1), 4)

        # Save evaluation as CSV
        with open(metrics_filepath, 'w') as f:
            dict_writer = csv.DictWriter(f, fieldnames=["question_id", "precision_all", "recall_all", "precision_col", "recall_col", "query_length", "order_correctness"])
            dict_writer.writeheader()
            dict_writer.writerows(metrics_store[1:] + [metrics_store[0]])

    def predict_alias_mappings(self, gold_aliases: str, candidate_aliases: str):
        mapping_prompt = self.output_mapping_prompt.format(gold_aliases=gold_aliases, candidate_aliases=candidate_aliases)
        self.logger.info(f"Prompting LLM using {mapping_prompt}")

        response = self.mapper_client.chat(prompt=mapping_prompt)
        response = response.replace("```json", "").replace("```", "").strip()
        self.logger.info(f"LLM response: {response}")

        response_object = json.loads(response)
        return response_object

    def filter_optional_columns(self, entry, optional_columns: list[str]):
        return {k: v for k, v in entry.items() if k not in optional_columns}

    def calc_row_metrics(self, gold_output: list[dict[str, any]], generated_output: list[dict[str, any]], eval_config: dict[str, any]):
        optional_columns = eval_config.get("optional_columns", [])
        order_config = eval_config.get("order")

        # Filter optional columns for structural matching
        gold_entries = [frozenset(self.filter_optional_columns(entry, optional_columns).items()) for entry in gold_output]
        generated_entries = [frozenset(self.filter_optional_columns(entry, optional_columns).items()) for entry in generated_output]

        # Include optional columns for value matching
        gold_entries_with_optional = [frozenset(entry.items()) for entry in gold_output]
        generated_entries_with_optional = [frozenset(entry.items()) for entry in generated_output]

        total_correct_pairs = 0
        total_predicted_pairs = 0
        total_gold_pairs = 0
        matched_generated_entries = set()

        # Find the best match from the gold entries
        for gold_entry, gold_entry_with_optional in zip(gold_entries, gold_entries_with_optional):
            best_match_i = None
            best_match_pairs = 0
            best_match_incorrect_optionals = 0

            for i, (generated_entry, generated_entry_with_optional) in enumerate(zip(generated_entries, generated_entries_with_optional)):
                if i in matched_generated_entries:
                    continue

                correct_pairs = gold_entry & generated_entry
                incorrect_optional_pairs = [
                    (k, v) for k, v in gold_entry_with_optional
                    if (k in optional_columns) and (k in dict(generated_entry_with_optional)) and (v != dict(generated_entry_with_optional).get(k))
                ]

                if len(correct_pairs) > best_match_pairs:
                    best_match_i = i
                    best_match_pairs = len(correct_pairs)
                    best_match_incorrect_optionals = len(incorrect_optional_pairs)

            total_correct_pairs += best_match_pairs
            total_gold_pairs += len(gold_entry) + best_match_incorrect_optionals
            total_predicted_pairs += len(generated_entry) + best_match_incorrect_optionals

            if best_match_i is not None:
                matched_generated_entries.add(best_match_i)

        # Handle unmatched predicted pairs
        total_predicted_pairs += sum(len(entry) for i, entry in enumerate(generated_entries) if i not in matched_generated_entries)

        # Check order
        order_correctness = 1.0
        if order_config:
            alias_name, direction = order_config
            generated_sorted = sorted(generated_output, key=lambda x: x.get(alias_name, None), reverse=(direction == "DESC"))
            order_correctness = 1.0 if all(output == output_sorted for output, output_sorted, in zip(generated_output, generated_sorted)) else 0.0

        # Calculate scores
        precision_all = total_correct_pairs / total_predicted_pairs if total_predicted_pairs > 0 else 0.0
        recall_all = total_correct_pairs / total_gold_pairs if total_gold_pairs > 0 else 0.0

        return {"precision_all": round(precision_all, 4), "recall_all": round(recall_all, 4), "order_correctness": order_correctness}

    def calc_col_metrics(self, gold_aliases: list[str], normalised_candidate_aliases: list[str], optional_columns: list[str]):
        gold_set = set([alias for alias in gold_aliases if alias not in optional_columns])
        candidate_set = set([alias for alias in normalised_candidate_aliases if alias not in optional_columns])

        true_positives = gold_set & candidate_set

        precision = len(true_positives) / len(candidate_set) if candidate_set else 0.0
        recall = len(true_positives) / len(gold_set) if gold_set else 0.0

        return {"precision_col": round(precision, 4), "recall_col": round(recall, 4)}

def test_row_metrics():
    evaluator = PlanOutputEvaluator(mapper_client=None)

    eval_config = {
        "optional_columns": "country",
        "order": None
    }

    # Exact same
    gold_output = [{"name": "Alice", "age": 30, "country": "A"}, {"name": "Bob", "age": 25, "country": "B"}]
    generated_output = [{"name": "Alice", "age": 30, "country": "A"}, {"name": "Bob", "age": 25, "country": "B"}]
    print(f"Exact same: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

    # Missing optional column
    generated_output = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    print(f"Missing optional column: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

    # Optional column with wrong values
    generated_output = [{"name": "Alice", "age": 30, "country": "B"}, {"name": "Bob", "age": 25, "country": "A"}]
    print(f"Optional column with wrong values: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

    # Missing row
    generated_output = [{"name": "Alice", "age": 30}]
    print(f"Missing row: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

    # Additional row
    generated_output = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}]
    print(f"Additional row: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

    # Wrong row with some same values
    generated_output = [{"name": "Alice", "age": 30}, {"name": "Charlie", "age": 25}]
    print(f"Wrong row with some same values: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

    # Missing column
    generated_output = [{"name": "Alice"}, {"name": "Bob"}]
    print(f"Missing column: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

    # Additional column
    generated_output = [{"name": "Alice", "age": 30, "city": "New York"}, {"name": "Bob", "age": 25, "city": "New York"}]
    print(f"Additional column: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

    # Bad order
    eval_config = {
        "optional_columns": "country",
        "order": ["age", "DESC"]
    }
    generated_output = [{"name": "Bob", "age": 25, "country": "B"}, {"name": "Alice", "age": 30, "country": "A"}]
    print(f"Bad order: {evaluator.calc_row_metrics(gold_output, generated_output, eval_config=eval_config)}")

def test_col_metrics():
    evaluator = PlanOutputEvaluator(mapper_client=None)
    optional_cols = ["extra"]

    # Exact same
    gold_aliases = ["name", "age", "city", "extra"]
    normalised_candidate_aliases = ["name", "age", "city", "extra"]
    print(f"Exact same: {evaluator.calc_col_metrics(gold_aliases, normalised_candidate_aliases, optional_cols)}")

    # Missing optional col
    normalised_candidate_aliases = ["name", "age", "city"]
    print(f"Missing optional field: {evaluator.calc_col_metrics(gold_aliases, normalised_candidate_aliases, optional_cols)}")

    # One wrong name
    normalised_candidate_aliases = ["name", "age", "country"]
    print(f"One wrong field: {evaluator.calc_col_metrics(gold_aliases, normalised_candidate_aliases, optional_cols)}")

    # One additional field
    normalised_candidate_aliases = ["name", "age", "city", "country"]
    print(f"One additional field: {evaluator.calc_col_metrics(gold_aliases, normalised_candidate_aliases, optional_cols)}")
