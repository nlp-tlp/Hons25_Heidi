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
        metrics_store = []
        for gold_query, generated_query in zip(gold_queries, generated_queries):
            entry_id = gold_query["id"]

            # Execute
            try:
                gold_output = plan_execution_function(gold_query["ground_truth_query"])
                generated_output = plan_execution_function(generated_query["generated_query"])
            except Exception as e:
                self.logger.error(f"Error during query/ plan execution on ID {entry_id}: {e}")
                break

            # Normalise generated output if aliases are different
            gold_field_names = list(gold_output[0].keys())
            generated_field_names = list(generated_output[0].keys())

            if not set(gold_field_names) == set(generated_field_names):
                field_mappings = self.predict_alias_mappings(gold_aliases=gold_field_names, candidate_aliases=generated_field_names)
                print(f"MAPPING: {field_mappings}")

                generated_output = [
                    {field_mappings.get(k, k): v for k, v in entry.items()} for entry in generated_output
                ]

            # Calculate and score evaluation
            metrics = {}
            metrics["question_id"] = entry_id
            metrics.update(self.calc_row_metrics(gold_output=gold_output, generated_output=generated_output))
            metrics.update(self.calc_col_metrics(gold_aliases=gold_field_names, normalised_candidate_aliases=list(generated_output[0].keys())))
            metrics_store.append(metrics)

        # Save evaluation as CSV
        with open(metrics_filepath, 'w') as f:
            dict_writer = csv.DictWriter(f, fieldnames=["question_id", "precision_all", "recall_all", "precision_col", "recall_col"])
            dict_writer.writeheader()
            dict_writer.writerows(metrics_store)

    def predict_alias_mappings(self, gold_aliases: str, candidate_aliases: str):
        mapping_prompt = self.output_mapping_prompt.format(gold_aliases=gold_aliases, candidate_aliases=candidate_aliases)
        self.logger.debug(f"Prompting LLM using {mapping_prompt}")

        response = self.mapper_client.chat(prompt=mapping_prompt)
        self.logger.debug(f"LLM response: {response}")

        response_object = json.load(response)
        return response_object

    def calc_row_metrics(self, gold_output: list[dict[str, any]], generated_output: list[dict[str, any]]):
        gold_entries = [frozenset(entry.items()) for entry in gold_output]
        generated_entries = [frozenset(entry.items()) for entry in generated_output]

        total_correct_pairs = 0
        total_predicted_pairs = 0
        total_gold_pairs = 0
        matched_generated_entries = set()

        # Find the best match from the gold entries
        for gold_entry in gold_entries:
            best_match_i = None
            best_match_pairs = 0
            max_pairs = gold_entry & gold_entry

            for i, generated_entry in enumerate(generated_entries):
                if i in matched_generated_entries:
                    continue

                correct_pairs = gold_entry & generated_entry
                if len(correct_pairs) > best_match_pairs:
                    best_match_i = i
                    best_match_pairs = len(correct_pairs)

                    if len(matched_generated_entries) == len(gold_entries) or best_match_pairs == max_pairs:
                        break

            total_correct_pairs += best_match_pairs
            total_gold_pairs += len(gold_entry)

            if best_match_i is not None:
                matched_generated_entries.add(best_match_i)

        # Calculate scores
        total_predicted_pairs += sum(len(entry) for entry in generated_entries)

        precision_all = total_correct_pairs / total_predicted_pairs if total_predicted_pairs > 0 else 0.0
        recall_all = total_correct_pairs / total_gold_pairs if total_gold_pairs > 0 else 0.0

        return {"precision_all": round(precision_all, 4), "recall_all": round(recall_all, 4)}

    def calc_col_metrics(self, gold_aliases: list[str], normalised_candidate_aliases: list[str]):
        gold_set = set(gold_aliases)
        candidate_set = set(normalised_candidate_aliases)

        true_positives = gold_set & candidate_set

        precision = len(true_positives) / len(candidate_set) if candidate_set else 0.0
        recall = len(true_positives) / len(gold_set) if gold_set else 0.0

        return {"precision_col": round(precision, 4), "recall_col": round(recall, 4)}

def test_row_metrics():
    evaluator = PlanOutputEvaluator(mapper_client=None)

    # Exact same
    gold_output = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    generated_output = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    print(f"Exact same: {evaluator.calc_row_metrics(gold_output, generated_output)}")

    # Missing row
    generated_output = [{"name": "Alice", "age": 30}]
    print(f"Missing row: {evaluator.calc_row_metrics(gold_output, generated_output)}")

    # Additional row
    generated_output = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}]
    print(f"Additional row: {evaluator.calc_row_metrics(gold_output, generated_output)}")

    # Wrong row with some same values
    generated_output = [{"name": "Alice", "age": 30}, {"name": "Charlie", "age": 25}]
    print(f"Wrong row with some same values: {evaluator.calc_row_metrics(gold_output, generated_output)}")

    # Missing column
    generated_output = [{"name": "Alice"}, {"name": "Bob"}]
    print(f"Missing column: {evaluator.calc_row_metrics(gold_output, generated_output)}")

    # Additional column
    generated_output = [{"name": "Alice", "age": 30, "city": "New York"}, {"name": "Bob", "age": 25, "city": "New York"}]
    print(f"Additional column: {evaluator.calc_row_metrics(gold_output, generated_output)}")

def test_col_metrics():
    evaluator = PlanOutputEvaluator(mapper_client=None)

    # Exact same
    gold_aliases = ["name", "age", "city"]
    normalised_candidate_aliases = ["name", "age", "city"]
    print(f"Exact same: {evaluator.calc_col_metrics(gold_aliases, normalised_candidate_aliases)}")

    # One wrong name
    normalised_candidate_aliases = ["name", "age", "country"]
    print(f"One wrong field: {evaluator.calc_col_metrics(gold_aliases, normalised_candidate_aliases)}")

    # One additional field
    normalised_candidate_aliases = ["name", "age", "city", "country"]
    print(f"One additional field: {evaluator.calc_col_metrics(gold_aliases, normalised_candidate_aliases)}")
