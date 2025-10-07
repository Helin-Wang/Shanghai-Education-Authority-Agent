import jieba
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, Chroma
from models.M3eEmbedding import M3eEmbeddings
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sqlite3
from database.utils import chroma_filter, chroma_search
from database.index import bm25_search

class HybridRetriever:
    """Hybrid retriever combining BM25 keyword search and FAISS semantic search"""
    
    def __init__(self, faiss_db_path: str = None, chroma_db_path: str = None, conn: sqlite3.Connection = None, k: int = 10):
        """
        Initialize hybrid retriever
        
        Args:
            docs: List of documents to index
            data_path: Path to JSON data file (alternative to docs)
            k: Number of documents to retrieve
        """
        self.k = k
        self.bm25_retriever = None
        self.faiss_db = None
        self.chroma_db = None
        self.embedding_model = M3eEmbeddings()
        self.conn = None
        
        # FAISS
        if faiss_db_path is not None:
            self._build_from_faiss(faiss_db_path)
        else:
            raise ValueError("faiss_db_path must be provided")
        
        # BM25
        if conn is not None:
            self.conn = conn
        else:
            raise ValueError("conn must be provided")
    
    def _build_from_faiss(self, faiss_db_path: str):
        """Build indexes from FAISS database"""
        self.faiss_db = FAISS.load_local(faiss_db_path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        # self.faiss_db.k = self.k
    
    def _build_from_chroma(self, chroma_db_path: str):
        """Build indexes from Chroma database"""
        self.chroma_db = Chroma(embedding_function=M3eEmbeddings(), persist_directory=chroma_db_path)

    
    def _build_from_conn(self, conn: sqlite3.Connection):
        """Build indexes from SQLite database"""
        cur = conn.cursor()
        cur.execute("""
            SELECT chunk_id, page_content, doc_id, doc_title, doc_link, category, year
            FROM v_chunks_join
            ORDER BY doc_id, chunk_id
        """)
        rows = cur.fetchall()
        
        docs = []
        for chunk_id, text, doc_id, doc_title, doc_link, category, year in rows:
            md = {
                "chunk_index": chunk_id,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_link": doc_link,
                "category": category,
                "year": year,
            }
            docs.append(Document(page_content=text or "", metadata=md))
            
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = self.k
    
    def retrieve(self, query: str, alpha: float = 0.5, years: List[str] = None, categories: List[str] = None, return_scores: bool = False) -> Union[List[Document], Tuple[List[Document], List[float]]]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Search query
            alpha: Weight for BM25 (1-alpha is weight for FAISS)
            years: Optional list of years to filter by
            categories: Optional list of categories to filter by
            return_scores: If True, return both documents and scores
            
        Returns:
            List of retrieved documents, or tuple of (documents, scores) if return_scores=True
        """
        def filter_docs(docs, years=None, categories=None):
            def valid(doc):
                if years and doc.metadata.get("year") not in years:
                    return False
                if categories and doc.metadata.get("category") not in categories:
                    return False
                return True

            return [(doc, score) for doc, score in docs if valid(doc)]

        # BM25 retrieval
        bm25_docs_with_scores = bm25_search(self.conn, query)
        # filter by years and categories
        bm25_docs_with_scores = filter_docs(bm25_docs_with_scores, years, categories)
        bm25_docs = [doc for doc, score in bm25_docs_with_scores]
        bm25_scores = [score for doc, score in bm25_docs_with_scores]
       # print(f"BM25 scores: {bm25_scores}")
        # FAISS retrieval: smaller scores are better
        faiss_docs_with_scores = self.faiss_db.similarity_search_with_score(query, k=self.k)
        # filter by years and categories
        faiss_docs_with_scores = filter_docs(faiss_docs_with_scores, years, categories)
        faiss_docs = [doc for doc, score in faiss_docs_with_scores]
        faiss_scores = [score for doc, score in faiss_docs_with_scores]
       # print(f"FAISS scores: {faiss_scores}")
        
        # Normalize scores
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        faiss_scores_norm = self._normalize_scores(faiss_scores)
        
        if return_scores:
            # Get scores for the combined results
            combined_results_with_scores = self._combine_results_with_scores(
                bm25_docs, bm25_scores_norm,
                faiss_docs, faiss_scores_norm,
                alpha
            )
            docs = [doc for doc, score in combined_results_with_scores[:self.k]]
            scores = [score for doc, score in combined_results_with_scores[:self.k]]
            return docs, scores
        else:
            # Combine results
            combined_results = self._combine_results(
                bm25_docs, bm25_scores_norm,
                faiss_docs, faiss_scores_norm,
                alpha
            )
            return combined_results[:self.k]
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _combine_results(self, bm25_docs: List[Document], bm25_scores: np.ndarray,
                        faiss_docs: List[Document], faiss_scores: List[float],
                        alpha: float) -> List[Document]:
        """Combine BM25 and FAISS results"""
        # Create document to score mapping
        doc_scores = {}
        
        # Add BM25 scores
        for i, doc in enumerate(bm25_docs):
            doc_key = self._get_doc_key(doc)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + alpha * bm25_scores[i]
        
        # Add FAISS scores
        for i, doc in enumerate(faiss_docs):
            doc_key = self._get_doc_key(doc)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + (1 - alpha) * faiss_scores[i]
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1])
        
        # Extract documents
        result_docs = []
        for doc_key, score in sorted_docs:
            # Find the document from either BM25 or FAISS results
            doc = self._find_doc_by_key(doc_key, bm25_docs + faiss_docs)
            if doc is not None:
                result_docs.append(doc)
        
        return result_docs
    
    def _combine_results_with_scores(self, bm25_docs: List[Document], bm25_scores: np.ndarray,
                                   faiss_docs: List[Document], faiss_scores: List[float],
                                   alpha: float) -> List[Tuple[Document, float]]:
        """Combine BM25 and FAISS results with scores"""
        # Create document to score mapping
        doc_scores = {}
        
        # Add BM25 scores
        for i, doc in enumerate(bm25_docs):
            doc_key = self._get_doc_key(doc)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + alpha * bm25_scores[i]
        
        # Add FAISS scores
        for i, doc in enumerate(faiss_docs):
            doc_key = self._get_doc_key(doc)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + (1 - alpha) * faiss_scores[i]
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1])
        
        # Extract documents with scores
        result_docs = []
        for doc_key, score in sorted_docs:
            # Find the document from either BM25 or FAISS results
            doc = self._find_doc_by_key(doc_key, bm25_docs + faiss_docs)
            if doc is not None:
                result_docs.append((doc, score))
        return result_docs
    
    def _get_doc_key(self, doc: Document) -> str:
        """Return chunk_index"""
        return doc.metadata.get('chunk_index', '')
    
    def _find_doc_by_key(self, doc_key: str, docs: List[Document]) -> Document:
        """Find a document by its key"""
        for doc in docs:
            if self._get_doc_key(doc) == doc_key:
                return doc
        return None
