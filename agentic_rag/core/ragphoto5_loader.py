"""
RAGPhoto_5 Component Loader
Safely loads components from RAGPhoto_5 without namespace conflicts

Strategy: Keep RAGPhoto_5 permanently in sys.path at position 1 (after current project)
This is the simplest and most reliable approach.
"""

import sys
import os
from pathlib import Path


class RAGPhoto5Loader:
    """Loader for RAGPhoto_5 components"""
    
    RAGPHOTO5_PATH = '/home/wzfeng/RAGPhoto/Data_Agentic_RAG/data_pipeline'
    _components_loaded = False
    _cached_components = None
    
    @classmethod
    def _ensure_path(cls):
        """Ensure RAGPhoto_5 is in sys.path at the right position"""
        # Add RAGPhoto_5 at position 1 (after current project at position 0)
        if cls.RAGPHOTO5_PATH not in sys.path:
            # Insert at position 1, not 0, so current project takes precedence
            sys.path.insert(1, cls.RAGPHOTO5_PATH)
        elif sys.path.index(cls.RAGPHOTO5_PATH) == 0:
            # If it's at position 0, move it to position 1
            sys.path.remove(cls.RAGPHOTO5_PATH)
            sys.path.insert(1, cls.RAGPHOTO5_PATH)
    
    @classmethod
    def load_components(cls):
        """
        Load and return RAGPhoto_5 components
        
        Returns:
            Tuple of (MilvusVectorStore, EmbeddingGenerator, LLMGenerator, 
                     Neo4jManager, RerankerModel classes)
        """
        # Return cached components if already loaded
        if cls._components_loaded and cls._cached_components:
            return cls._cached_components
        
        try:
            # Ensure RAGPhoto_5 is in path
            cls._ensure_path()
            
            # Now import using standard import
            # Python will find modules in RAGPhoto_5 first due to path order
            from core.vector_store import MilvusVectorStore
            from core.embeddings import EmbeddingGenerator
            from core.llm import LLMGenerator
            from core.neo4j_manager import Neo4jManager
            
            # Try to load Reranker (optional)
            try:
                from core.reranker import RerankerModel
            except Exception:
                RerankerModel = None
            
            # Cache the components
            cls._cached_components = (
                MilvusVectorStore,
                EmbeddingGenerator,
                LLMGenerator,
                Neo4jManager,
                RerankerModel
            )
            cls._components_loaded = True
            
            return cls._cached_components
            
        except Exception as e:
            raise ImportError(f"Failed to load RAGPhoto_5 components: {e}")
    
    @classmethod
    def check_availability(cls):
        """Check if RAGPhoto_5 is available"""
        return Path(cls.RAGPHOTO5_PATH).exists()

