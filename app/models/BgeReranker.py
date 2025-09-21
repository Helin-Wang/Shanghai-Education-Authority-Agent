import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from typing import List, Tuple
from langchain.schema import Document

class BgeReranker:
    """BGE Reranker for reranking retrieved documents"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the BGE reranker
        
        Args:
            model_path: Path to the BGE reranker model. If None, uses default path.
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "bge-reranker-base")
        
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            print(f"BGE Reranker model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading BGE reranker model: {e}")
            raise e
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: The search query
            documents: List of Document objects to rerank
            top_k: Number of top documents to return. If None, returns all documents.
            
        Returns:
            List of reranked Document objects
        """
        if not documents:
            return []
        
        if len(documents) == 1:
            return documents
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            pairs.append([query, doc.page_content])
        
        # Get scores
        scores = self._compute_scores(pairs)
        
        # Create list of (document, score) tuples
        doc_scores = list(zip(documents, scores))
        
        # Sort by score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract reranked documents
        reranked_docs = [doc for doc, score in doc_scores]
        
        # Return top_k documents if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
        
        return reranked_docs
    
    def _compute_scores(self, pairs: List[List[str]]) -> List[float]:
        """
        Compute relevance scores for query-document pairs
        
        Args:
            pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores
        """
        with torch.no_grad():
            # Tokenize pairs
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Extract scores (logits)
            scores = outputs.logits.squeeze(-1)
            
            # Convert to probabilities using sigmoid
            scores = torch.sigmoid(scores)
            
            return scores.cpu().numpy().tolist()
    
    def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with scores
        
        Args:
            query: The search query
            documents: List of Document objects to rerank
            top_k: Number of top documents to return. If None, returns all documents.
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        if len(documents) == 1:
            return [(documents[0], 1.0)]
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            pairs.append([query, doc.page_content])
        
        # Get scores
        scores = self._compute_scores(pairs)
        
        # Create list of (document, score) tuples
        doc_scores = list(zip(documents, scores))
        
        # Sort by score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents if specified
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
