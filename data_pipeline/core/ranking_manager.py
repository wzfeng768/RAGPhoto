"""
排序管理器
统一管理相似度排序和重排序功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Tuple
from config.config import config
from .reranker import DocumentReranker

class RankingManager:
    """排序管理器"""
    
    def __init__(self):
        """初始化排序管理器"""
        # 初始化重排序器
        self.document_reranker = DocumentReranker()
        
        print("初始化排序管理器:")
        print(f"  相似度排序: 始终可用")
        print(f"  重排序功能: {'启用' if config.enable_reranking else '禁用'}")
        if config.enable_reranking:
            print(f"  重排序模型: {config.reranker_model}")
    
    def rank_documents(self, 
                      query: str, 
                      documents: List[Dict[str, Any]], 
                      top_k: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        排序文档（相似度排序或重排序）
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的文档数量
            
        Returns:
            (排序后的文档列表, 排序信息)
        """
        if not documents:
            return documents, {
                'method': 'none',
                'model': None,
                'reranked': False
            }
        
        # 如果启用了重排序，使用重排序器
        if self.document_reranker.is_enabled():
            return self.document_reranker.rerank_documents(query, documents, top_k)
        else:
            # 否则使用相似度排序（已经按相似度排序）
            limited_docs = documents[:top_k] if top_k else documents
            return limited_docs, {
                'method': 'similarity',
                'model': None,
                'reranked': False
            }
    
    def compare_ranking_methods(self, 
                               query: str, 
                               documents: List[Dict[str, Any]], 
                               top_k: int = 5) -> Dict[str, Any]:
        """
        比较不同排序方法的结果
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 比较的文档数量
            
        Returns:
            比较结果
        """
        print(f"\n🔍 比较排序方法 (前{top_k}个文档)")
        print("=" * 60)
        
        # 相似度排序结果
        similarity_docs = documents[:top_k]
        
        # 重排序结果（临时启用）
        original_state = self.document_reranker.enabled
        self.document_reranker.enabled = True
        reranked_docs, rerank_info = self.document_reranker.rerank_documents(
            query, documents, top_k
        )
        self.document_reranker.enabled = original_state
        
        # 比较结果
        print("\n📊 相似度排序 vs 重排序对比:")
        print("-" * 40)
        
        for i in range(min(top_k, len(similarity_docs), len(reranked_docs))):
            sim_doc = similarity_docs[i]
            rerank_doc = reranked_docs[i] if i < len(reranked_docs) else None
            
            print(f"\n第{i+1}位:")
            print(f"  相似度排序: ...{sim_doc.get('content', '')[:50]}...")
            print(f"  相似度分数: {sim_doc.get('score', 0.0):.4f}")
            
            if rerank_doc:
                print(f"  重排序结果: ...{rerank_doc.get('content', '')[:50]}...")
                print(f"  重排序分数: {rerank_doc.get('rerank_score', 0.0):.4f}")
            
            # 检查是否是同一文档
            same_doc = (sim_doc.get('content') == rerank_doc.get('content')) if rerank_doc else False
            print(f"  是否相同: {'✓' if same_doc else '✗'}")
        
        print("-" * 40)
        
        return {
            'similarity_ranking': similarity_docs,
            'reranked_results': reranked_docs,
            'rerank_info': rerank_info,
            'query': query
        }
    
    def toggle_reranking(self) -> bool:
        """
        切换重排序功能
        
        Returns:
            当前重排序状态
        """
        return self.document_reranker.toggle_reranking()
    
    def print_ranking_status(self):
        """打印当前排序状态"""
        print("\n📊 排序功能状态:")
        print("=" * 30)
        print(f"相似度排序: ✓ 启用")
        print(f"重排序功能: {'✓ 启用' if self.document_reranker.is_enabled() else '✗ 禁用'}")
        
        if self.document_reranker.is_enabled():
            print(f"重排序模型: {config.reranker_model}")
        
        print("=" * 30)
    
    def get_ranking_info(self) -> Dict[str, Any]:
        """
        获取排序配置信息
        
        Returns:
            排序配置字典
        """
        return {
            'similarity_enabled': True,
            'reranking_enabled': self.document_reranker.is_enabled(),
            'reranker_model': config.reranker_model if config.enable_reranking else None,
            'reranker_top_k': config.reranker_top_k
        } 