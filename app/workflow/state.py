from typing import List, Optional, Dict, Any, TypedDict
from langchain.schema import Document

class AgentState(TypedDict):
    """State for the LangGraph workflow"""
    query: str
    docs: List[Document]
    reranked_docs: List[Document]  # Reranked documents after reranking step
    history: List[Dict[str, str]]  # [{"role":"user","content":"..."}, ...]
    answer: Optional[str]
    
    conn: Optional[Any]  # Store connection instance
    faiss_db_path: Optional[Any]  # Store FAISS database path
    retriever: Optional[Any]  # Store retriever instance
    reranker: Optional[Any]  # Store reranker instance
    llm: Optional[Any]  # Store LLM instance
    years: Optional[List[str]]  # Store years
    categories: Optional[List[str]]  # Store categories