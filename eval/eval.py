import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))
from agents.graph import retriever_test_workflow_app
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qapair_filepath", help="QAPair Filepath", default="../data/eval/simpleQA_v0.json")
    args = parser.parse_args()
    qapair_filepath = args.qapair_filepath
    
    # Load qapairs
    with open(qapair_filepath, "r", encoding="utf-8") as f:
        qapairs = json.load(f)
    
    for qapair in qapairs:
        print(qapair)
        question = qapair["question"]
        result_state = retriever_test_workflow_app.invoke({"query": question})
        # print the retrieved docs
        print(result_state)
        break
    # result_state = retriever_test_workflow_app.invoke({"query": qapairs[0]["question"]})
    # # print the retrieved docs
    # print(result_state["docs"])
    # print(result_state["reranked_docs"])