"""
Configuration Management for Agentic RAG v2
Loads settings from .env file and provides typed access to configuration values
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Configuration class for Agentic RAG v2"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        # Load .env file from project root
        # Note: override=True ensures agentic_rag/.env values take precedence
        # over any previously loaded env vars (e.g., from evaluation/.env)
        project_root = Path(__file__).parent.parent
        env_file = project_root / '.env'
        if env_file.exists():
            load_dotenv(env_file, override=True)
        
        # ========================================
        # OpenAI API Configuration
        # ========================================
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required but not set in .env file")
        
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # LLM Configuration
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        
        # Embedding Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
        
        # Embedding API (optional: defaults to OpenAI if not set)
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.openai_api_key)
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", self.openai_base_url)
        
        # ========================================
        # Milvus Configuration
        # ========================================
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19531"))  # 修正端口: 19530 -> 19531
        self.milvus_collection = os.getenv("MILVUS_COLLECTION", "academic_papers")  # 修正集合名
        self.milvus_index_type = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
        self.milvus_metric_type = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
        
        # Compatibility alias for RAGPhoto_5
        self.collection_name = self.milvus_collection
        self.vector_dimension = self.embedding_dimensions
        
        # ========================================
        # Neo4j Configuration
        # ========================================
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # ========================================
        # Agent Configuration
        # ========================================
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "3"))
        self.quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.7"))
        self.enable_early_stopping = os.getenv("ENABLE_EARLY_STOPPING", "true").lower() == "true"
        self.early_stop_threshold = float(os.getenv("EARLY_STOP_THRESHOLD", "0.9"))
        self.verbose = os.getenv("VERBOSE", "false").lower() == "true"
        
        # ========================================
        # Retrieval Configuration
        # ========================================
        self.vector_top_k = int(os.getenv("VECTOR_TOP_K", "20"))  # Increased from 15 to 20
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))
        
        # Compatibility alias for RAGPhoto_5
        self.top_k_results = self.vector_top_k
        
        # Reranking
        self.enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.reranker_model = os.getenv("RERANKER_MODEL", "qwen3-reranker-8b")
        self.reranker_top_n = int(os.getenv("RERANKER_TOP_N", "8"))  # Increased from 5 to 8
        self.reranker_api_key = os.getenv("RERANKER_API_KEY", self.openai_api_key)
        self.reranker_base_url = os.getenv("RERANKER_BASE_URL", "https://www.dmxapi.cn/v1")
        
        # Compatibility alias for RAGPhoto_5
        self.reranker_top_k = self.reranker_top_n
        
        # Knowledge Graph
        self.kg_max_entities = int(os.getenv("KG_MAX_ENTITIES", "10"))
        self.kg_max_relationships = int(os.getenv("KG_MAX_RELATIONSHIPS", "20"))
        self.kg_collection_name = os.getenv("KG_COLLECTION_NAME", "photovoltaic_kg")
        self.kg_vector_dimension = int(os.getenv("KG_VECTOR_DIMENSION", "1536"))
        
        # ========================================
        # Answer Validation (NEW)
        # ========================================
        self.enable_answer_validation = os.getenv("ENABLE_ANSWER_VALIDATION", "true").lower() == "true"
        self.validation_auto_correct = os.getenv("VALIDATION_AUTO_CORRECT", "true").lower() == "true"
        
        # ========================================
        # Retrieval Precision (NEW)
        # ========================================
        self.min_similarity_threshold = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.30"))
        self.min_rerank_score = float(os.getenv("MIN_RERANK_SCORE", "0.5"))
        self.dynamic_retrieval_params = os.getenv("DYNAMIC_RETRIEVAL_PARAMS", "true").lower() == "true"
        
        # Section Prioritization (NEW)
        self.section_priorities = {
            'numerical': ['results', 'performance', 'experimental'],
            'method': ['methodology', 'experimental', 'fabrication'],
            'general': ['introduction', 'results', 'discussion']
        }
        
        # ========================================
        # Web Search Configuration
        # ========================================
        self.enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
        self.web_search_results = int(os.getenv("WEB_SEARCH_RESULTS", "5"))
        self.web_search_engine = os.getenv("WEB_SEARCH_ENGINE", "duckduckgo")
        self.web_search_mode = os.getenv("WEB_SEARCH_MODE", "both")
        self.web_search_start_iteration = int(os.getenv("WEB_SEARCH_START_ITERATION", "1"))
        self.web_enabled_sources = os.getenv(
            "WEB_ENABLED_SOURCES",
            "arxiv,pubmed,semantic_scholar,core"
        )
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.core_api_key = os.getenv("CORE_API_KEY", "")
        self.pubmed_email = os.getenv("PUBMED_EMAIL", "user@example.com")
        self.web_search_timeout = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
        
        # ========================================
        # Performance Optimization
        # ========================================
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_size = int(os.getenv("CACHE_SIZE", "1000"))
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        self.parallel_tool_execution = os.getenv("PARALLEL_TOOL_EXECUTION", "true").lower() == "true"
        
        # ========================================
        # Data Source and Text Processing
        # ========================================
        self.data_source = os.getenv("DATA_SOURCE", "/data/wzfeng/RAGPhoto/Data/AI_MDs")
        self.project_root = os.getenv("PROJECT_ROOT", "/data/wzfeng/RAGPhoto/Data_Agentic_RAG/agentic_rag")
        self.test_mds_path = self.data_source
        
        # Text Splitting Configuration (for RAGPhoto_5 compatibility)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Performance Optimization (for RAGPhoto_5 compatibility)
        self.enable_parallel_processing = os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        self.skip_duplicate_content = os.getenv("SKIP_DUPLICATE_CONTENT", "true").lower() == "true"
        self.min_chunk_length = int(os.getenv("MIN_CHUNK_LENGTH", "20"))
        
        # ========================================
        # Logging
        # ========================================
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "logs/agentic_rag.log")
        
        # ========================================
        # RAGPhoto_5 Integration Path
        # ========================================
        self.ragphoto5_path = os.getenv(
            "RAGPHOTO5_PATH", 
            "/data/wzfeng/RAGPhoto/Data_Agentic_RAG/data_pipeline"
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self) -> str:
        """String representation (hide sensitive keys)"""
        config_dict = self.to_dict()
        # Hide sensitive information
        if 'openai_api_key' in config_dict:
            config_dict['openai_api_key'] = '***' + config_dict['openai_api_key'][-4:]
        if 'neo4j_password' in config_dict:
            config_dict['neo4j_password'] = '***'
        return f"Config({config_dict})"


# Global configuration instance
config: Optional[Config] = None


def load_config(env_file: Optional[str] = None) -> Config:
    """
    Load configuration from .env file
    
    Args:
        env_file: Optional path to .env file (default: project_root/.env)
    
    Returns:
        Config instance
    """
    global config
    
    if env_file:
        load_dotenv(env_file)
    
    config = Config()
    return config


def get_config() -> Config:
    """
    Get global configuration instance
    
    Returns:
        Config instance (loads if not already loaded)
    """
    global config
    if config is None:
        config = Config()
    return config


# Auto-load configuration on module import
try:
    config = Config()
except ValueError as e:
    # Allow import even if configuration is incomplete
    # Useful for documentation generation and testing
    print(f"⚠️  Configuration warning: {e}")
    print("   Please copy config/env_example.txt to .env and set OPENAI_API_KEY")
    config = None


# Convenience function for backward compatibility with RAGPhoto_5
def load_env_file(config_file: str = ".env"):
    """Load environment file (for compatibility with RAGPhoto_5)"""
    if os.path.exists(config_file):
        load_dotenv(config_file)
    return get_config()

