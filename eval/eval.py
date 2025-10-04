import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))
from agents.graph import workflow_app
import argparse
import json
import sqlite3
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qapair_filepath", help="QAPair Filepath", default="../data/eval/official_faq.csv")
    parser.add_argument("--results_filepath", help="QAPair Filepath", default="./results/official_faq_v0.json")
    args = parser.parse_args()
    qapair_filepath = args.qapair_filepath
    results_filepath = args.results_filepath
    
    print("Loading QA Pairs...")
    print(qapair_filepath)
    # Load qapairs
    qapairs = []
    if qapair_filepath.endswith('.json'):
        print("JSON")
        with open(qapair_filepath, "r", encoding="utf-8") as f:
            qapairs = json.load(f)
    else:
        df = pd.read_csv(qapair_filepath)
        for index, row in df.iterrows():
            qapair = {
                "question": row['问题'],
                'answer': row['答案']
            }
            qapairs.append(qapair)
        
    eval_results = []
    conn = sqlite3.connect("../data/shanghai_education_authority_agent.db")
    for qapair in tqdm(qapairs):
        question = qapair["question"]
        initial_state = {
            "query": question,
            "docs": [],
            "history": [],
            "answer": None,
            "retriever": None,
            "llm": None,
            "conn": conn,
            "faiss_db_path": "../data/faiss_index"
        }
        result_state = workflow_app.invoke(initial_state)
        
        eval_results.append({
            "question": question,
            "reranked_docs": ",".join([doc.metadata['chunk_index'] for doc in result_state["reranked_docs"]]),
            "answer": result_state["answer"],
            "ground_truth": qapair["answer"]
        })
    conn.close()
    
    # create results directory if not exists
    if not os.path.exists("./results"):
        os.makedirs("./results")
    
    with open(results_filepath, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)