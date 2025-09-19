from typing import List, Optional, Dict, Any, TypedDict
from langchain.schema import Document

class AgentState(TypedDict):
    """State for the LangGraph workflow"""
    query: str
    docs: List[Document]
    reranked_docs: List[Document]  # Reranked documents after reranking step
    history: List[Dict[str, str]]  # [{"role":"user","content":"..."}, ...]
    answer: Optional[str]
    
    retriever: Optional[Any]  # Store retriever instance
    reranker: Optional[Any]  # Store reranker instance
    llm: Optional[Any]  # Store LLM instance
