import os
from app.workflow.graph import workflow_app
import sqlite3
# Initialize API configuration
api_key_r1 = 'ebe4d4b6-00ae-4ea7-9890-9356d6a29570'
os.environ["OPENAI_API_BASE"] = 'https://ark.cn-beijing.volces.com/api/v3'
os.environ["OPENAI_API_KEY"] = api_key_r1

def run_langgraph_workflow(query: str):
    """Run the LangGraph workflow for a given query"""
    
    # Initialize state
    conn = sqlite3.connect("../data/shanghai_education_authority_agent.db")
    initial_state = {
        "query": query,
        "docs": [],
        "history": [],
        "answer": None,
        "retriever": None,
        "llm": None,
        "conn": conn,
        "faiss_db_path": "../data/faiss_index"
    }
    
    # Run the workflow
    result = workflow_app.invoke(initial_state)
    conn.close()
    
    return result

if __name__ == "__main__":
    # Example query
    query = "什么时候公布学业水平考试的成绩？"
    
    print(f"Running LangGraph workflow for query: {query}")
    print("=" * 50)
    
    # Run the workflow
    result = run_langgraph_workflow(query)
    
    print("Retrieved Documents:")
    for i, doc in enumerate(result["docs"]):
        print(f"Document {i+1}:")
        print(doc)
        # print(f"  Title: {doc.metadata.get('title', 'N/A')}")
        # print(f"  Year: {doc.metadata.get('year', 'N/A')}")
        # print(f"  Content preview: {doc.page_content[:100]}...")
        print()
    
    print("Generated Answer:")
    print(result["answer"])