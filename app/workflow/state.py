from typing import List, Optional, Dict, Any, TypedDict
from langchain.schema import Document
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """State for the LangGraph workflow"""
    query: str
    answer: Optional[str]
    
    docs: List[Document]
    reranked_docs: List[Document]  # Reranked documents after reranking step
    history: List[BaseMessage]
    
    conn: Optional[Any]  # Store connection instance
    faiss_db_path: Optional[Any]  # Store FAISS database path
    chroma_db_path: Optional[Any]  # Store Chroma database path
    retriever: Optional[Any]  # Store retriever instance
    reranker: Optional[Any]  # Store reranker instance
    llm: Optional[Any]  # Store LLM instance
    years: Optional[List[str]]  # Store years
    categories: Optional[List[str]]  # Store categories
    
    # Clarification-related fields
    needs_clarification: Optional[bool]  # Whether the query needs clarification
    
    # HyDE (Hypothetical Document Embeddings) switch
    use_hyde: Optional[bool]  # Whether to use HyDE for retrieval enhancement