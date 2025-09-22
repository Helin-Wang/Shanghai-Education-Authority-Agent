import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))
from agents.graph import workflow_app
import argparse
import json
import sqlite3
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qapair_filepath", help="QAPair Filepath", default="../data/eval/simpleQA_v0.json")
    args = parser.parse_args()
    qapair_filepath = args.qapair_filepath
    
    # Load qapairs
    with open(qapair_filepath, "r", encoding="utf-8") as f:
        qapairs = json.load(f)
    
    eval_results = []
    for qapair in tqdm(qapairs):
        question = qapair["question"]
        initial_state = {
            "query": question,
            "docs": [],
            "history": [],
            "answer": None,
            "retriever": None,
            "llm": None,
            "conn": sqlite3.connect("../data/shanghai_education_authority_agent.db"),
            "faiss_db_path": "../data/faiss_index"
        }
        result_state = workflow_app.invoke(initial_state)
        # print the retrieved docs
        
        eval_results.append({
            "question": question,
            "reranked_docs": ",".join([doc.metadata['doc_id'] for doc in result_state["reranked_docs"]]),
            "answer": result_state["answer"],
            "ground_truth": qapair["answer"]
        })
   
    # create results directory if not exists
    if not os.path.exists("./results"):
        os.makedirs("./results")
    
    with open("./results/eval_results_v0.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)