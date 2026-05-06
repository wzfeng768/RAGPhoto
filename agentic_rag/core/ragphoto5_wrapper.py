"""
Direct wrapper to initialize RAGPhoto_5 components
This file should be run with RAGPhoto_5 in sys.path
"""

import sys
from pathlib import Path

# Add data_pipeline to path BEFORE any other imports
RAGPHOTO5_PATH = '/home/wzfeng/RAGPhoto/Data_Agentic_RAG/data_pipeline'
if RAGPHOTO5_PATH not in sys.path:
    sys.path.insert(0, RAGPHOTO5_PATH)


def create_vector_store():
    """Create and return MilvusVectorStore instance"""
    from core.vector_store import MilvusVectorStore
    return MilvusVectorStore()


def create_embedding_generator():
    """Create and return EmbeddingGenerator instance"""
    from core.embeddings import EmbeddingGenerator
    return EmbeddingGenerator()


def create_llm_generator():
    """Create and return LLMGenerator instance"""
    from core.llm import LLMGenerator
    return LLMGenerator()


def create_neo4j_manager():
    """Create and return Neo4jManager instance"""
    from core.neo4j_manager import Neo4jManager
    return Neo4jManager()


def create_reranker():
    """Create and return RerankerModel instance (or None if not available)"""
    try:
        from core.reranker import RerankerModel
        return RerankerModel()
    except Exception:
        return None

