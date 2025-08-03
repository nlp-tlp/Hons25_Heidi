import logging

from llm import ChatClient, EmbeddingClient
from retrievers import TextToCypherRetriever
from evaluation import PlanOutputEvaluator, test_row_metrics, test_col_metrics
from linking import EntityLinker

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

MODEL_ANSWERS_FILEPATH = "evaluation/model_answers/extended_cypher_model_answers.json"
GENERATED_ANSWERS_FILEPATH = "evaluation/experiment_runs/extended_cypher_run_t3.json"
METRICS_STORE_FILEPATH = "evaluation/experiment_runs/extended_cypher_run_t3metrics.csv"

def eval_t2ce(generate: bool = True, evaluate: bool = True,):
    # Initialise
    chat_client = ChatClient(provider="openai", model="gpt-4.1-2025-04-14")
    embedder_client = EmbeddingClient(provider="openai", model="text-embedding-3-small")
    t2c_retriever = TextToCypherRetriever(client=chat_client, prompt_path="retrievers/cypher/extended_cypher_prompt_fewshot.txt", embedding_client=embedder_client)

    # RAG component functions
    linker = EntityLinker(client=ChatClient(provider="openai", model="gpt-4.1-mini-2025-04-14"))
    def generator_function(question: str) -> str:
        extra_context = linker.get_linked_context(question=question)
        query = t2c_retriever.generate_cypher(question=question, extra_context=extra_context)
        return query

    def executor_function(query: str) -> list[dict[str, any]]:
        _original_query, output, _error = t2c_retriever.execute_query(query=query, extended_cypher=True)

        if _error:
            raise Exception()

        return output

    # Evaluate
    evaluator = PlanOutputEvaluator(mapper_client=chat_client)
    if generate:
        evaluator.generate(
            plan_generation_function=generator_function,
            model_answers_filepath=MODEL_ANSWERS_FILEPATH,
            generated_answers_filepath=GENERATED_ANSWERS_FILEPATH
        )
    if evaluate:
        evaluator.evaluate(
            plan_execution_function=executor_function,
            model_answers_filepath=MODEL_ANSWERS_FILEPATH,
            generated_answers_filepath=GENERATED_ANSWERS_FILEPATH,
            metrics_filepath=METRICS_STORE_FILEPATH
        )

if __name__ == "__main__":
    eval_t2ce(generate=True, evaluate=True)
    # test_row_metrics()
    # test_col_metrics()