import logging
import sys

from scopes import retriever_factory
from evaluation import QASet

logging.basicConfig(
    level=logging.INFO,
    format="\n=== %(levelname)s [%(name)s] ===\n%(message)s\n"
)

if __name__ == "__main__":
    if not len(sys.argv) == 3 or len(sys.argv) == 4:
        print("Incorrect number of arguments")
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
            qa_set.run_extract_nuggets()
        case "rag":
            run_file_path = f"evaluation/experiment_runs/{strategy}{"_link" if allow_linking else ""}.xlsx"
            qa_set.run_rag(retriever, run_file_path)
        case "eval":
            run_file_path = f"evaluation/experiment_runs/{strategy}{"_link" if allow_linking else ""}.xlsx"
            qa_set.run_match_nuggets(run_file_path)
        case _:
            print("Unrecognised action")
            exit(1)
