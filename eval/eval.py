import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))
print(sys.path)
from agents.graph import retriever_test_workflow_app

if __name__ == "__main__":
    result_state = retriever_test_workflow_app.invoke({"query": "什么时候公布学业水平考试的成绩？"})
    # print the retrieved docs
    print(result_state["docs"])
    print(result_state["reranked_docs"])