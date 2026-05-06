"""
Reranker Module using qwen3-reranker-8b
Uses API-based reranking for improved document relevance scoring
"""

import os
from typing import List, Dict, Any
from openai import OpenAI
from config.config import config


class Reranker:
    """
    Reranker using qwen3-reranker-8b model via API
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize Reranker
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.model = config.reranker_model
        self.top_n = config.reranker_top_n
        
        # Initialize OpenAI-compatible client for reranker
        self.client = OpenAI(
            api_key=config.reranker_api_key,
            base_url=config.reranker_base_url
        )
        
        if self.verbose:
            print(f"✅ Reranker initialized: {self.model}")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using qwen3-reranker-8b
        
        Args:
            query: User query
            documents: List of document dictionaries with 'content' field
            top_n: Number of top documents to return
            
        Returns:
            Reranked list of documents with added 'rerank_score' field
        """
        if not documents:
            return []
        
        # Validate documents is a list
        if not isinstance(documents, list):
            if self.verbose:
                print(f"   ⚠️  documents is not a list: {type(documents)}")
            return []
        
        # Filter out non-dict items
        valid_documents = []
        for doc in documents:
            if isinstance(doc, dict):
                valid_documents.append(doc)
            else:
                if self.verbose:
                    print(f"   ⚠️  Skipping non-dict document: {type(doc)}")
        
        if not valid_documents:
            if self.verbose:
                print(f"   ⚠️  No valid documents to rerank")
            return []
        
        documents = valid_documents
        top_n = top_n or self.top_n
        
        try:
            # Prepare documents for reranking
            doc_texts = [doc.get('content', '') for doc in documents]
            
            # Call reranker API
            # qwen3-reranker uses a special format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self._format_rerank_request(query, doc_texts)
                    }
                ],
                temperature=0.0,
                max_tokens=1000
            )
            
            # Parse scores from response
            scores = self._parse_rerank_scores(response, len(documents))
            
            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = score
            
            # Sort by rerank score (descending)
            reranked_docs = sorted(
                documents,
                key=lambda x: x.get('rerank_score', 0.0),
                reverse=True
            )
            
            # Return top N
            result = reranked_docs[:top_n]
            
            if self.verbose:
                print(f"   Reranked {len(documents)} → {len(result)} documents")
                if result:
                    print(f"   Top score: {result[0].get('rerank_score', 0.0):.3f}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Reranking failed: {e}")
            # Fallback: return original top_n documents with preserved similarity scores
            # Ensure documents have at least the original similarity score
            for doc in documents:
                if 'rerank_score' not in doc:
                    # Use original similarity if available
                    doc['rerank_score'] = doc.get('similarity', 0.0)
            return documents[:top_n]
    
    def _format_rerank_request(self, query: str, documents: List[str]) -> str:
        """
        Format rerank request for qwen3-reranker
        
        The model expects a specific format to understand it's a reranking task
        """
        request = f"Query: {query}\n\nRank the following documents by relevance:\n\n"
        
        for i, doc in enumerate(documents, 1):
            # Truncate long documents
            doc_text = doc[:500] if len(doc) > 500 else doc
            request += f"Document {i}:\n{doc_text}\n\n"
        
        request += "Return relevance scores (0-1) for each document as a JSON array."
        
        return request
    
    def _parse_rerank_scores(self, response, num_docs: int) -> List[float]:
        """
        Parse rerank scores from model response
        
        Args:
            response: API response
            num_docs: Expected number of documents
            
        Returns:
            List of scores (0-1)
        """
        try:
            import json
            import re
            
            content = response.choices[0].message.content
            
            # Try to extract JSON array
            json_match = re.search(r'\[[\d\.,\s]+\]', content)
            if json_match:
                scores = json.loads(json_match.group())
                
                # Normalize to 0-1 if needed
                if scores and max(scores) > 1:
                    max_score = max(scores)
                    scores = [s / max_score for s in scores]
                
                # Ensure correct length
                if len(scores) == num_docs:
                    return scores
            
            # Fallback: extract individual numbers
            numbers = re.findall(r'0\.\d+|\d+\.\d+', content)
            if numbers:
                scores = [float(n) for n in numbers[:num_docs]]
                if len(scores) == num_docs:
                    return scores
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Score parsing failed: {e}")
        
        # Ultimate fallback: uniform decent scores to indicate uncertainty
        # Use 0.5-0.6 range to show these are estimated
        return [0.6 - (i * 0.05) for i in range(min(num_docs, 10))] + [0.4] * max(0, num_docs - 10)


# Singleton instance
_reranker_instance = None


def get_reranker(verbose: bool = False) -> Reranker:
    """Get or create reranker singleton"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(verbose=verbose)
    return _reranker_instance

