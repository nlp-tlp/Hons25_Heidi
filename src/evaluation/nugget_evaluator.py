import logging
import pandas as pd
import tiktoken
import json
from typing import List, Literal
from pydantic import BaseModel

from llm import ChatClient
from generators import FinalGenerator

QA_PATH = "evaluation/model_answers.xlsx"
NUGGET_EXTRACTION_PROMPT = "evaluation/nugget_extraction_prompt.txt"
NUGGET_MATCHING_PROMPT = "evaluation/nugget_matching_prompt.txt"

class Nugget(BaseModel):
    nugget: str
    status: Literal["ESSENTIAL", "OPTIONAL"]

class NuggetExtractionResponse(BaseModel):
    nuggets: List[Nugget]

class NuggetMatch(BaseModel):
    nugget: str
    status: Literal["ESSENTIAL", "OPTIONAL"]
    match: Literal["MATCHED", "MISSING", "INCORRECT"]

class NuggetMatchingResponse(BaseModel):
    nugget_results: List[NuggetMatch]
    extra_claims: List[str]

class QASet:
    def __init__(self, nugget_extraction_prompt_path=NUGGET_EXTRACTION_PROMPT, nugget_matching_prompt=NUGGET_MATCHING_PROMPT):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.chat_client = ChatClient(provider="openai", model="gpt-4.1-2025-04-14")
        self.generator = FinalGenerator(client=ChatClient(provider="openai", model="gpt-4.1-2025-04-14"))

        with open(nugget_extraction_prompt_path, 'r') as f:
            self.nugget_extraction_prompt = f.read()
        with open(nugget_matching_prompt, 'r') as f:
            self.nugget_matching_prompt = f.read()

    def run_rag(self, retriever, run_file_path: str, model_answers_path=QA_PATH):
        df_model = pd.read_excel(model_answers_path).to_dict(orient="records")

        rag_run = []
        for entry in df_model:
            question_id = entry["ID"]
            question = entry["Question"]

            cypher_query, retrieved_records, error = retriever.retrieve(question)
            if error:
                rag_run.append({"ID": question_id, "Question": question, "Model_Answer": entry["Answer"], "Query": cypher_query, "Final_Response": f"EXECUTION ERROR: {error}", "Retrieved_Tok_Length": 0})
                continue
            final_response = self.generator.generate(question=question, retrieved_nodes=retrieved_records, schema_context=retriever.schema_context())

            retrieved_records_str = "\n".join([str(r) for r in retrieved_records])
            retrieved_records_length = self.metric_tok_length(retrieved_records_str)

            rag_run.append({"ID": question_id, "Question": question, "Model_Answer": entry["Answer"], "Query": cypher_query, "Final_Response": final_response, "Retrieved_Tok_Length": retrieved_records_length})

        df = pd.DataFrame(rag_run)
        df.to_excel(run_file_path, index=False)

    def metric_tok_length(self, text: str):
        enc = tiktoken.get_encoding("o200k_base") # tokeniser for gpt-4.1
        return len(enc.encode(text))

    def run_extract_nuggets(self, model_answers_path=QA_PATH):
        df_model = pd.read_excel(model_answers_path).to_dict(orient="records")

        extracted = []
        for entry in df_model:
            question = entry["Question"]
            model_answer = entry["Answer"]

            prompt = self.nugget_extraction_prompt.format(
                question=question,
                model_answer=model_answer
            )
            self.logger.info(f"Prompting LLM using: {prompt}")
            response = self.chat_client.chat(prompt=prompt, response_format=NuggetExtractionResponse)
            self.logger.info(f"Received response: {response}")

            # Process response
            response_json = json.loads(response)
            entry["Nuggets"] = json.dumps(response_json["nuggets"], ensure_ascii=False)
            extracted.append(entry)

        df_new = pd.DataFrame(extracted)
        df_new.to_excel(model_answers_path, index=False)

    def run_match_nuggets(self, run_file_path: str, model_answers_path=QA_PATH):
        df_run = pd.read_excel(run_file_path).to_dict(orient="records")
        df_model = pd.read_excel(model_answers_path).to_dict(orient="records")

        results = []
        for run_entry, model_entry in zip(df_run, df_model):
            question = model_entry["Question"]
            candidate_answer = run_entry["Final_Response"]

            if candidate_answer[:15] == "EXECUTION ERROR":
                results.append({
                    "Nugget_Results": None,
                    "Extra_Claims": None,
                    "Precision": 0,
                    "Recall": 0
                })
                continue

            prompt = self.nugget_matching_prompt.format(
                question=question,
                model_nuggets=model_entry["Nuggets"],
                generated_answer=candidate_answer
            )
            self.logger.info(f"Prompting LLM using: {prompt}")
            response = self.chat_client.chat(prompt=prompt, response_format=NuggetMatchingResponse)
            self.logger.info(f"Received response: {response}")

            # Process response
            response_json = json.loads(response)
            precision, recall = self.nugget_metrics(response_json["nugget_results"], response_json["extra_claims"])

            results.append({
                "Nugget_Results": json.dumps(response_json["nugget_results"], ensure_ascii=False),
                "Extra_Claims": json.dumps(response_json["extra_claims"], ensure_ascii=False),
                "Precision": precision,
                "Recall": recall
            })

        df_results = pd.DataFrame(results)
        df_final = pd.concat([pd.DataFrame(df_run), df_results], axis=1)
        df_final.to_excel(run_file_path, index=False)

    # For running after human nugget evaluation fixing
    def run_metrics_only(self, run_file_path: str):
        df_run = pd.read_excel(run_file_path).to_dict(orient="records")

        metrics = []
        for run_entry in df_run:
            nuggets = run_entry["Nugget_Results"]
            extra = run_entry["Extra_Claims"]

            if not nuggets or not isinstance(nuggets, str):
                metrics.append({
                    "Precision": 0,
                    "Recall": 0
                })
                continue

            precision, recall = self.nugget_metrics(json.loads(nuggets), json.loads(extra))
            metrics.append({
                "Precision": precision,
                "Recall": recall
            })

        df_metrics = pd.DataFrame(metrics)
        df_final = pd.concat([pd.DataFrame(df_run), df_metrics], axis=1)
        df_final.to_excel(run_file_path, index=False)

    def nugget_metrics(self, nugget_results, extra_claims, optional_weight=0.3):
        # Counts
        essential = [n for n in nugget_results if n["status"] == "ESSENTIAL"]
        optional = [n for n in nugget_results if n["status"] == "OPTIONAL"]

        matched_essential = sum(1 for n in essential if n["match"] == "MATCHED")
        matched_optional = sum(1 for n in optional if n["match"] == "MATCHED")

        # Recall - ESSENTIAL nuggets + penalise for incorrect OPTIONAL matches
        incorrect_optional = sum(1 for n in optional if n["match"] == "INCORRECT")
        recall = matched_essential / (len(essential) + incorrect_optional) if essential else 1.0

        # Precision - ESSENTIAL matches + weighted OPTIONAL matches
        total_matched = matched_essential + optional_weight * matched_optional
        total_system = matched_essential + optional_weight * matched_optional + optional_weight * len(extra_claims)
        precision = total_matched / total_system if total_system > 0 else 0.0

        return round(precision, 4), round(recall, 4)
