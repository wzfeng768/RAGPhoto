"""
RAG系统核心模块

包含RAG系统的所有核心功能组件：
- 文档处理和分割
- 向量生成和存储
- 智能检索和排序
- LLM生成和管理
"""

from .document_loader import DocumentLoader
from .photovoltaic_text_splitter import PhotovoltaicTextSplitter
from .embeddings import EmbeddingGenerator
from .vector_manager import VectorManager
from .rag_system import RAGSystem
from .knowledge_graph import PhotovoltaicKnowledgeGraph, Entity, Relation, EntityType, RelationType
from .neo4j_manager import Neo4jManager
from .hybrid_retriever import HybridRetriever
from .kg_enhanced_rag import KGEnhancedRAGSystem
from .question_loader import QuestionLoader
from .entity_normalizer import EntityNormalizer, get_normalizer

__all__ = [
    # 主要系统
    'RAGSystem',
    'KGEnhancedRAGSystem',
    
    # 文本处理
    'PhotovoltaicTextSplitter',
    'DocumentLoader',
    
    # 向量处理
    'EmbeddingGenerator',
    'MilvusVectorStore',
    'VectorManager',
    
    # 生成和排序
    'LLMGenerator',
    'RerankerModel',
    'DocumentReranker',
    'RankingManager',
    
    # 知识图谱
    'PhotovoltaicKnowledgeGraph',
    'Entity',
    'Relation',
    'EntityType',
    'RelationType',
    'Neo4jManager',
    'HybridRetriever',
    'EntityNormalizer',
    'get_normalizer',
    
    # 数据加载
    'QuestionLoader',
]

# 版本信息
__version__ = '2.1.0'
__author__ = 'RAGPhoto Team'
__description__ = '光电科研文献RAG系统核心模块（实体合并+缩写识别）'
