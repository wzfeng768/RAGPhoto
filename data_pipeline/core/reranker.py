"""
重排序模块
使用外部重排序模型对检索结果进行重新排序
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from typing import List, Dict, Any, Tuple
from config.config import config

class RerankerModel:
    """重排序模型"""
    
    def __init__(self, 
                 model_name: str = None,
                 api_key: str = None,
                 base_url: str = None):
        """
        初始化重排序模型
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            base_url: API基础URL
        """
        self.model_name = model_name or config.reranker_model
        self.api_key = api_key or config.reranker_api_key
        self.base_url = base_url or config.reranker_base_url
        
        print("初始化重排序模型:")
        print(f"  模型: {self.model_name}")
        print(f"  API地址: {self.base_url}")
        print(f"  API Key: {'已设置' if self.api_key else '未设置'}")
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的文档数量
            
        Returns:
            (原始索引, 重排序分数) 的列表，按分数降序排列
        """
        return self._api_rerank(query, documents, top_k)
    

    
    def _api_rerank(self, query: str, documents: List[str], top_k: int = None, max_retries: int = 2) -> List[Tuple[int, float]]:
        """
        使用API进行重排序（带重试机制）
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的文档数量
            max_retries: 最大重试次数
            
        Returns:
            API重排序结果
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # 构建API请求
                url = f"{self.base_url.rstrip('/')}/rerank"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k or len(documents)
                }
                
                # 发送请求（增加超时时间到60秒）
                response = requests.post(url, json=data, headers=headers, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                
                # 解析结果
                scores = []
                if "results" in result:
                    for item in result["results"]:
                        original_index = item.get("index", 0)
                        score = item.get("relevance_score", 0.0)
                        scores.append((original_index, score))
                
                return scores
                
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"⚠️ 重排序API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"   等待后重试...")
                    import time
                    time.sleep(2)  # 等待2秒后重试
                else:
                    print(f"⚠️ 重排序API调用失败 (已重试{max_retries}次): {e}")
        
        # 如果所有重试都失败，返回原始顺序
        return [(i, 1.0 - i * 0.1) for i in range(len(documents))]

class DocumentReranker:
    """文档重排序器"""
    
    def __init__(self):
        """初始化文档重排序器"""
        self.reranker_model = RerankerModel(
            model_name=config.reranker_model,
            api_key=config.reranker_api_key,
            base_url=config.reranker_base_url
        )
        
        self.enabled = config.enable_reranking
        
        if self.enabled:
            print(f"✅ 重排序功能已启用")
        else:
            print("❌ 重排序功能已禁用")
    
    def rerank_documents(self, 
                        query: str, 
                        documents: List[Dict[str, Any]], 
                        top_k: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 原始文档列表
            top_k: 返回的文档数量
            
        Returns:
            (重排序后的文档列表, 排序信息)
        """
        if not self.enabled or not documents:
            return documents, {
                'method': 'similarity',
                'model': None,
                'reranked': False
            }
        
        try:
            # 提取文档内容用于重排序
            doc_contents = [doc.get('content', '') for doc in documents]
            
            print(f"🔄 使用 {self.reranker_model.model_name} 重排序 {len(documents)} 个文档...")
            
            # 调用重排序模型
            rerank_results = self.reranker_model.rerank(
                query=query,
                documents=doc_contents,
                top_k=top_k or config.reranker_top_k
            )
            
            # 根据重排序结果重新排列文档
            reranked_docs = []
            for original_idx, rerank_score in rerank_results:
                if original_idx < len(documents):
                    doc = documents[original_idx].copy()
                    doc['rerank_score'] = rerank_score
                    reranked_docs.append(doc)
            
            ranking_info = {
                'method': 'reranking',
                'model': self.reranker_model.model_name,
                'reranked': True,
                'original_count': len(documents),
                'reranked_count': len(reranked_docs)
            }
            
            print(f"✅ 重排序完成，返回 {len(reranked_docs)} 个文档")
            
            return reranked_docs, ranking_info
            
        except Exception as e:
            print(f"❌ 重排序失败: {e}")
            # 如果重排序失败，返回原始结果
            return documents, {
                'method': 'similarity_fallback',
                'model': None,
                'reranked': False,
                'error': str(e)
            }
    
    def toggle_reranking(self) -> bool:
        """
        切换重排序功能
        
        Returns:
            当前重排序状态
        """
        self.enabled = not self.enabled
        status = "启用" if self.enabled else "禁用"
        print(f"🔄 重排序功能已{status}")
        return self.enabled
    
    def is_enabled(self) -> bool:
        """
        检查重排序是否启用
        
        Returns:
            重排序状态
        """
        return self.enabled 