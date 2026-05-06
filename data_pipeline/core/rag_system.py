"""
RAG系统主类
整合向量检索、重排序和LLM生成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
from config.config import config, load_env_file
from .vector_store import MilvusVectorStore
from .embeddings import EmbeddingGenerator
from .llm import LLMGenerator
from .photovoltaic_text_splitter import PhotovoltaicTextSplitter

class RAGSystem:
    """RAG系统主类"""
    
    def __init__(self, config_file: str = ".env", verbose: bool = True):
        """
        初始化RAG系统
        
        Args:
            config_file: 配置文件路径
            verbose: 是否显示详细输出
        """
        self._verbose = verbose
        
        if self._verbose:
            print("=" * 60)
            print("初始化光电科研文献RAG系统")
            print("=" * 60)
        
        # 加载配置文件
        load_env_file(config_file)
        
        # 初始化组件
        self.text_splitter = PhotovoltaicTextSplitter(
            default_chunk_size=config.chunk_size,
            default_chunk_overlap=config.chunk_overlap
        )
        if self._verbose:
            print("✓ 使用光电科研文献智能分块器")
        
        self.embedding_generator = EmbeddingGenerator()
        self.llm_generator = LLMGenerator()
        
        # 初始化向量存储
        self.vector_store = None
        
        # 初始化向量管理器
        from .vector_manager import VectorManager
        self.vector_manager = VectorManager()
        
        # 初始化排序管理器
        from .ranking_manager import RankingManager
        self.ranking_manager = RankingManager()
        
        if self._verbose:
            print("✓ RAG系统初始化完成")
    
    def initialize_vector_store(self):
        """初始化向量存储（Milvus服务应该已通过docker-compose启动）"""
        if not self.vector_store:
            # 初始化向量存储
            self.vector_store = MilvusVectorStore()
            # MilvusVectorStore会自动连接到已有的collection
            # 如果collection不存在，会在运行pipeline时创建
            if self._verbose:
                print("✓ 向量存储已初始化")
    
    def rag_query(self, query: str, top_k: Optional[int] = None, return_context: bool = False) -> Dict[str, Any]:
        """
        执行RAG查询
        
        Args:
            query: 用户查询
            top_k: 检索文档数量
            return_context: 是否返回上下文信息（用于评估）
            
        Returns:
            查询结果字典
        """
        import time
        start_time = time.time()
        
        top_k = top_k or config.top_k_results
        
        if self._verbose:
            print(f"\n执行RAG查询: {query}")
            print(f"检索文档数: {top_k}")
        
        # 1. 确保向量存储已初始化
        if not self.vector_store:
            self.initialize_vector_store()
        
        # 2. 向量检索
        if self._verbose:
            print("正在检索相关文档...")
        similar_docs = self.search(query, top_k)
        
        if not similar_docs:
            return {
                'success': False,
                'error': 'No relevant documents found',
                'query': query,
                'retrieved_docs': [],
                'retrieved_docs_count': 0,
                'answer': 'No relevant documents found to answer the query.',
                'total_time': 0,
                'query_time': 0
            }
        
        if self._verbose:
            print(f"找到 {len(similar_docs)} 个相关文档")
        
        # 3. 文档重排序
        ranked_docs, ranking_info = self.ranking_manager.rank_documents(
            query, similar_docs
        )
        
        if self._verbose:
            if ranking_info['method'] == 'reranking':
                print(f"📊 重排序完成，使用 {ranking_info['model']} 模型")
            else:
                print(f"📊 相似度排序完成")
        
        # 3. 生成回答
        if self._verbose:
            print("正在生成回答...")
        llm_result = self.llm_generator.rag_query(query, ranked_docs)
        
        # 构建返回结果
        end_time = time.time()
        total_query_time = end_time - start_time
        
        result = {
            'success': llm_result.get('success', False),
            'query': query,
            'answer': llm_result.get('answer', ''),
            'retrieved_docs': ranked_docs,  # 返回文档列表
            'retrieved_docs_count': len(similar_docs),  # 文档数量
            'ranking_method': ranking_info['method'],
            'tokens_used': llm_result.get('tokens_used'),
            'total_time': total_query_time,  # 总查询时间
            'query_time': total_query_time,  # 兼容旧字段名
            'generation_time': llm_result.get('generation_time', 0)  # LLM生成时间
        }
        
        # 如果需要返回上下文（用于评估）
        if return_context:
            result['context_documents'] = [
                {
                    'content': doc['content'],
                    'source': doc.get('source', ''),
                    'score': doc.get('score', 0.0)
                }
                for doc in ranked_docs
            ]
            result['contexts'] = [doc['content'] for doc in ranked_docs]
        
        if llm_result.get('error'):
            result['error'] = llm_result['error']
        
        return result
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        向量检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # 向量检索
            results = self.vector_store.search(query_embedding, top_k)
            
            # 处理结果格式
            processed_results = []
            for result in results:
                processed_result = {
                    'content': result['content'],
                    'source': result['source'],
                    'score': result['score'],
                    'chunk_id': result['chunk_id'],
                    'metadata': result['metadata']
                }
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            print(f"检索失败: {e}")
            return []
    
    def print_rag_result(self, result: Dict[str, Any]):
        """
        打印RAG查询结果
        
        Args:
            result: RAG查询结果
        """
        if not result.get('success'):
            print(f"❌ 查询失败: {result.get('error', '未知错误')}")
            return
        
        print(f"\n{'='*60}")
        print(f"问题: {result['query']}")
        print(f"{'='*60}")
        print(f"\n💡 回答:")
        print("-" * 50)
        print(result['answer'])
        print("-" * 50)
        
        print(f"\n📊 检索信息:")
        print(f"  检索到的文档数: {result['retrieved_docs']}")
        print(f"  排序方法: {result['ranking_method']}")
        if result.get('tokens_used'):
            print(f"  使用的tokens: {result['tokens_used']}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            系统统计信息
        """
        stats = {
            'milvus_running': self.milvus_manager.check_service_running(),
            'vector_store_initialized': self.vector_store is not None
        }
        
        if self.vector_store:
            stats.update(self.vector_store.get_collection_stats())
        
        # 添加向量管理器统计
        vector_stats = self.vector_manager.get_stats()
        stats.update(vector_stats)
        
        return stats
    
    def print_system_stats(self):
        """打印系统统计信息"""
        stats = self.get_system_stats()
        
        print("\n📊 系统状态:")
        print("=" * 50)
        print(f"Milvus服务: {'✓ 运行中' if stats.get('milvus_running') else '✗ 未运行'}")
        print(f"向量存储: {'✓ 已初始化' if stats.get('vector_store_initialized') else '✗ 未初始化'}")
        
        if stats.get('total_entities', 0) > 0:
            print(f"文档数量: {stats.get('total_entities', 0)}")
            print(f"向量维度: {stats.get('vector_dimension', 0)}")
            print(f"索引状态: {'✓ 已创建' if stats.get('has_index') else '✗ 未创建'}")
        
        # 显示向量管理器统计
        if stats.get('vector_built'):
            print(f"向量数据: ✓ 已构建 ({stats.get('build_time', 'N/A')})")
        else:
            print(f"向量数据: ✗ 未构建")
        
        print("=" * 50) 