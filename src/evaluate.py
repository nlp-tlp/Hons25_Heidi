import logging
import sys

from scopes import retriever_factory, retriever_choices
from evaluation import QASet

logging.basicConfig(
    level=logging.CRITICAL,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

if __name__ == "__main__":
    # For looping through all evaluation options
    if len(sys.argv) == 2 and sys.argv[1] == "loop_all":
        qa_set = QASet()
        for choice in retriever_choices:
            strategy = choice["name"]
            allow_linking = choice.get("allow_linking", False)
            retriever = retriever_factory(strategy, allow_linking)
            if not retriever:
                print(f"Unrecognised strategy: {strategy}")
                continue
            print(f"Running RAG for strategy: {strategy}, entity linking: {allow_linking}")
            run_file_path = f"evaluation/experiment_runs/{strategy}{'_link' if allow_linking else ''}.xlsx"
            qa_set.run_rag(retriever, run_file_path)
            print(f"Running evaluation for strategy: {strategy}, entity linking: {allow_linking}")
            qa_set.run_match_nuggets(run_file_path)
        exit(0)

    if not len(sys.argv) == 3 and not len(sys.argv) == 4:
        print(f"Incorrect number of arguments: {len(sys.argv)}")
        exit(1)

    strategy = sys.argv[1]
    allow_linking = True if len(sys.argv) == 4 else False
    retriever = retriever_factory(strategy, allow_linking)
    if not retriever:
        print("Unrecognised strategy")
        exit(1)

    qa_set = QASet()

    action = sys.argv[2]
    match action:
        case "nugget":
            print("Running nugget extraction for model answers.")
            qa_set.run_extract_nuggets()
        case "rag":
            print(f"Running RAG run for strategy: {strategy}, entity linking: {allow_linking}")
            run_file_path = f"evaluation/experiment_runs/{strategy}{"_link" if allow_linking else ""}.xlsx"
            qa_set.run_rag(retriever, run_file_path)
        case "eval":
            print(f"Running evaluation of RAG run for strategy: {strategy}, entity linking: {allow_linking}")
            run_file_path = f"evaluation/experiment_runs/{strategy}{"_link" if allow_linking else ""}.xlsx"
            qa_set.run_match_nuggets(run_file_path)
        case "metric":
            print(f"Running metric calculation of created nuggets run for strategy: {strategy}, entity linking: {allow_linking}")
            run_file_path = f"evaluation/experiment_runs/{strategy}{"_link" if allow_linking else ""}.xlsx"
            qa_set.run_metrics_only(run_file_path)
        case _:
            print("Unrecognised action")
            exit(1)
