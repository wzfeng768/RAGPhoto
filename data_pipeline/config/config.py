"""
配置管理模块
统一管理RAG系统的所有配置参数
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """系统配置类"""
    
    # 基础路径配置
    project_root: str = "/data/wzfeng/RAGPhoto/Data_Agentic_RAG/data_pipeline"
    # test_mds_path: str = "/data/wzfeng/RAGPhoto/Data/Test_MDs"
    # test_mds_path: str = "/data/wzfeng/RAGPhoto/Data/AI_MDs"
    test_mds_path: str = "/data/wzfeng/RAGPhoto/MDs"  # 使用完整数据
    
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19531  # 使用新端口避免与RAGPhoto_5冲突
    collection_name: str = "academic_papers"  # 实际集合名 (修正: test_academic_papers -> academic_papers)
    
    # 知识图谱配置
    kg_collection_name: str = "photovoltaic_kg"  # 知识图谱向量集合
    kg_vector_dimension: int = 1536  # 实体向量维度
    
    # Neo4j配置
    neo4j_uri: str = "bolt://localhost:7688"  # 使用新端口避免冲突
    neo4j_user: str = "neo4j"
    neo4j_password: str = "testpassword"  # Test password
    neo4j_database: str = "neo4j"  # Use default database
    
    # 向量配置
    vector_dimension: int = 1536  # text-embedding-3-small模型的维度（更快）
    top_k_results: int = 12  # 优化检索数量：提高精度 (15->12)
    similarity_threshold: float = 0.30  # 提高相似度阈值：过滤低质量结果 (0.15->0.30)
    
    # 文本分割配置
    chunk_size: int = 800  # 减小分块大小：1000->800
    chunk_overlap: int = 200  # 增加重叠：100->200
    
    # OpenAI API配置
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    embedding_batch_size: int = 2000  # 大幅增加批处理大小：100->1000

    # Embedding专用API配置 (可独立于LLM API)
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    
    # 性能优化配置
    enable_parallel_processing: bool = True  # 启用并行处理
    max_workers: int = 20  # 并行工作线程数
    skip_duplicate_content: bool = True  # 跳过重复内容
    min_chunk_length: int = 20  # 最小分块长度，跳过过短内容
    
    # LLM生成配置
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    
    # 重排序配置
    enable_reranking: bool = True
    reranker_model: str = "qwen3-reranker-8b"
    reranker_top_k: int = 5
    min_rerank_score: float = 0.5  # 最小重排序分数阈值
    reranker_api_key: str = ""
    reranker_base_url: str = "https://www.dmxapi.cn/v1"
    
    def __post_init__(self):
        """初始化后处理"""
        self.load_from_env()
    
    def load_from_env(self):
        """从环境变量加载配置"""
        # OpenAI配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", self.openai_base_url)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.llm_model = os.getenv("LLM_MODEL", self.llm_model)

        # Embedding专用API配置 (默认使用OpenAI配置)
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.openai_api_key)
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", self.openai_base_url)
        
        # 重排序配置
        self.enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.reranker_model = os.getenv("RERANKER_MODEL", self.reranker_model)
        self.reranker_api_key = os.getenv("RERANKER_API_KEY", self.openai_api_key)
        self.reranker_base_url = os.getenv("RERANKER_BASE_URL", self.openai_base_url)
        
        # Milvus配置
        self.milvus_host = os.getenv("MILVUS_HOST", self.milvus_host)
        self.milvus_port = int(os.getenv("MILVUS_PORT", str(self.milvus_port)))
        self.collection_name = os.getenv("COLLECTION_NAME", self.collection_name)
        
        # Neo4j配置
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.neo4j_database = os.getenv("NEO4J_DATABASE", self.neo4j_database)
        
        # 验证必需的配置
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            print("⚠️  OPENAI_API_KEY 未设置或使用默认值，请在.env文件中配置实际的API密钥")
            print("   某些功能可能无法正常工作")
    
    def print_config(self):
        """打印配置信息"""
        print("\n📋 系统配置:")
        print("=" * 50)
        print(f"文档路径: {self.test_mds_path}")
        print(f"Milvus服务: {self.milvus_host}:{self.milvus_port}")
        print(f"集合名称: {self.collection_name}")
        print(f"向量维度: {self.vector_dimension}")
        print(f"检索数量: {self.top_k_results}")
        print(f"嵌入模型: {self.embedding_model}")
        print(f"语言模型: {self.llm_model}")
        print(f"重排序功能: {'启用' if self.enable_reranking else '禁用'}")
        if self.enable_reranking:
            print(f"重排序模型: {self.reranker_model}")
        print("=" * 50)

def load_env_file(env_file: str = ".env"):
    """加载.env文件"""
    try:
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            print(f"已加载配置文件: {env_file}")
        else:
            print(f"配置文件 {env_file} 不存在")
    except Exception as e:
        print(f"加载配置文件失败: {e}")

# 自动加载.env文件并创建全局配置实例
load_env_file()
config = Config()
# 在创建config实例后重新加载环境变量
config.load_from_env() 