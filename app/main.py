import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.M3eEmbedding import M3eEmbeddings
from langchain_openai import ChatOpenAI
import os
from openai import OpenAI
from workflow.graph import workflow_app

# Initialize API configuration
api_key_r1 = 'ebe4d4b6-00ae-4ea7-9890-9356d6a29570'
os.environ["OPENAI_API_BASE"] = 'https://ark.cn-beijing.volces.com/api/v3'
os.environ["OPENAI_API_KEY"] = api_key_r1

def run_langgraph_workflow(query: str):
    """Run the LangGraph workflow for a given query"""
    
    # Initialize state
    initial_state = {
        "query": query,
        "docs": [],
        "history": [],
        "answer": None,
        "retriever": None,
        "llm": None
    }
    
    # Run the workflow
    result = workflow_app.invoke(initial_state)
    
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
        print(f"  Title: {doc.metadata.get('title', 'N/A')}")
        print(f"  Year: {doc.metadata.get('year', 'N/A')}")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print()
    
    print("Generated Answer:")
    print(result["answer"])