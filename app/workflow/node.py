from typing import Dict, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from models.M3eEmbedding import M3eEmbeddings
from models.BgeReranker import BgeReranker
from models.HybridRetriever import HybridRetriever
import os
from workflow.state import AgentState

# Initialize API configuration
api_key = 'sk-hmqokjrhfszsquludqhbdzftggjriimfelvjjqwzccxnqxmn'
os.environ["OPENAI_API_BASE"] = 'https://api.siliconflow.cn/v1'
os.environ["OPENAI_API_KEY"] = api_key

def retrieve_node(state: AgentState) -> AgentState:
    """根据查询从不同数据库中检索相关文档 - 使用混合检索方法"""
    query = state["query"]
    
    # Initialize hybrid retriever if not already done
    if "retriever" not in state or state["retriever"] is None:
        # Use hybrid retriever combining BM25 and FAISS
        hybrid_retriever = HybridRetriever(
            data_path="../data/v0_html_content.json",
            k=6  # Retrieve more documents initially, will be filtered by reranker
        )
        state["retriever"] = hybrid_retriever
    
    # Retrieve documents using hybrid approach
    retriever = state["retriever"]
    docs = retriever.retrieve(query, alpha=0.1)  # 0.1 for BM25, 0.9 for FAISS
    state["docs"] = docs
    
    return state

def rerank_node(state: AgentState) -> AgentState:
    """Rerank retrieved documents using BGE reranker"""
    query = state["query"]
    docs = state["docs"]
    
    # Initialize reranker if not already done
    if "reranker" not in state or state["reranker"] is None:
        reranker = BgeReranker()
        state["reranker"] = reranker
    
    # Rerank documents
    reranker = state["reranker"]
    reranked_docs = reranker.rerank(query, docs, top_k=3)  # Keep top 3 after reranking
    state["reranked_docs"] = reranked_docs
    
    return state

def generate_node(state: AgentState) -> AgentState:
    """Generate answer based on reranked documents"""
    query = state["query"]
    docs = state["reranked_docs"]  # Use reranked documents instead of original docs
    
    # Initialize LLM if not already done
    if "llm" not in state or state["llm"] is None:
        llm = ChatOpenAI(
            model='Qwen/Qwen2.5-7B-Instruct', 
            openai_api_key=api_key,
            openai_api_base='https://api.siliconflow.cn/v1',
            streaming=True
        )
        state["llm"] = llm
    
    # Create prompt with retrieved documents
    prompt = f"""
    You are a helpful assistant that can answer questions about Shanghai Education Authority information.
    Based on the following retrieved documents:
    {docs}
    
    Question: {query}
    
    Please provide a comprehensive answer based on the retrieved information. If the information is not sufficient, please indicate that.
    
    Answer:
    """
    
    # Generate response
    llm = state["llm"]
    response = llm.invoke(prompt)
    state["answer"] = response.content
    
    return state
