"""
嵌入向量生成模块
使用OpenAI API生成文本的向量表示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import numpy as np
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Dict, Tuple
from config.config import config

class EmbeddingGenerator:
    """嵌入向量生成器"""
    
    def __init__(self):
        """初始化嵌入向量生成器"""
        print("初始化嵌入向量生成器:")
        print(f"  模型: {config.embedding_model}")
        print(f"  批处理大小: {config.embedding_batch_size}")
        if config.enable_parallel_processing:
            print(f"  并行线程数: {config.max_workers}")
        
        # 初始化OpenAI客户端 (使用embedding专用配置)
        self.client = openai.OpenAI(
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url
        )
        
        # 内容缓存，避免重复计算
        self.content_cache: Dict[str, List[float]] = {}
        
    def _get_content_hash(self, text: str) -> str:
        """获取文本内容的哈希值用于缓存"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        return text.replace("\n", " ").strip()
    
    def _is_valid_content(self, text: str) -> bool:
        """检查内容是否有效（长度、非空等）"""
        return len(text.strip()) >= config.min_chunk_length
    
    def generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        生成单个文本的嵌入向量
        
        Args:
            text: 输入文本
            is_query: 是否为查询文本（查询文本允许更短）
            
        Returns:
            嵌入向量列表
        """
        # 预处理和验证
        text = self._preprocess_text(text)
        if not is_query and not self._is_valid_content(text):
            raise ValueError(f"文本内容过短或无效: {len(text)} 字符")
        elif is_query and len(text.strip()) < 2:
            raise ValueError(f"查询文本过短: {len(text)} 字符")
        
        # 检查缓存
        if config.skip_duplicate_content:
            text_hash = self._get_content_hash(text)
            if text_hash in self.content_cache:
                return self.content_cache[text_hash]
        
        try:
            response = self.client.embeddings.create(
                model=config.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # 缓存结果
            if config.skip_duplicate_content:
                self.content_cache[text_hash] = embedding
                
            return embedding
        except Exception as e:
            print(f"生成嵌入向量时出错: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量（优化版本）
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表的列表
        """
        start_time = time.time()
        
        # 预处理和过滤
        valid_texts = []
        text_indices = []  # 记录有效文本在原列表中的索引
        cached_embeddings = {}  # 缓存的embedding
        
        for i, text in enumerate(texts):
            processed_text = self._preprocess_text(text)
            
            # 跳过无效内容
            if not self._is_valid_content(processed_text):
                continue
                
            # 检查缓存
            if config.skip_duplicate_content:
                text_hash = self._get_content_hash(processed_text)
                if text_hash in self.content_cache:
                    cached_embeddings[i] = self.content_cache[text_hash]
                    continue
            
            valid_texts.append(processed_text)
            text_indices.append(i)
        
        print(f"  过滤后有效文本: {len(valid_texts)}/{len(texts)} (缓存命中: {len(cached_embeddings)})")
        
        # 如果没有需要处理的新文本，直接返回缓存结果
        if not valid_texts:
            result = [None] * len(texts)
            for idx, embedding in cached_embeddings.items():
                result[idx] = embedding
            return [emb for emb in result if emb is not None]
        
        # 生成embedding
        embeddings = []
        batch_size = config.embedding_batch_size
        
        if config.enable_parallel_processing and len(valid_texts) > batch_size:
            # 并行处理大批次
            embeddings = self._generate_embeddings_parallel(valid_texts, batch_size)
        else:
            # 串行处理
            embeddings = self._generate_embeddings_serial(valid_texts, batch_size)
        
        # 合并缓存结果和新生成的结果
        result = []
        new_embedding_idx = 0
        
        for i, text in enumerate(texts):
            if i in cached_embeddings:
                result.append(cached_embeddings[i])
            elif i in text_indices:
                if new_embedding_idx < len(embeddings):
                    embedding = embeddings[new_embedding_idx]
                    result.append(embedding)
                    
                    # 更新缓存
                    if config.skip_duplicate_content:
                        processed_text = self._preprocess_text(text)
                        text_hash = self._get_content_hash(processed_text)
                        self.content_cache[text_hash] = embedding
                    
                    new_embedding_idx += 1
        
        elapsed = time.time() - start_time
        print(f"  批量处理完成: {len(result)}个向量，用时 {elapsed:.2f}秒")
        
        return result
    
    def _generate_embeddings_serial(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """串行生成embeddings"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=config.embedding_model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                print(f"  已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文本")
                
            except Exception as e:
                print(f"批量生成嵌入向量时出错: {e}")
                raise
        
        return embeddings
    
    def _generate_embeddings_parallel(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """并行生成embeddings"""
        embeddings = [None] * len(texts)
        
        # 创建批次
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append((i, batch))
        
        print(f"  并行处理 {len(batches)} 个批次...")
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch_idx, batch): (start_idx, batch)
                for start_idx, batch in batches
                for batch_idx in [start_idx // batch_size]
            }
            
            completed = 0
            for future in as_completed(future_to_batch):
                start_idx, batch = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    # 将结果放入正确位置
                    for j, embedding in enumerate(batch_embeddings):
                        embeddings[start_idx + j] = embedding
                    
                    completed += len(batch)
                    print(f"  已完成 {completed}/{len(texts)} 个文本")
                    
                except Exception as e:
                    print(f"并行批次处理失败: {e}")
                    raise
        
        # 过滤None值（虽然不应该有）
        return [emb for emb in embeddings if emb is not None]
    
    def _process_batch(self, batch_idx: int, batch: List[str]) -> List[List[float]]:
        """处理单个批次"""
        try:
            response = self.client.embeddings.create(
                model=config.embedding_model,
                input=batch
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"批次 {batch_idx} 处理失败: {e}")
            raise
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.content_cache),
            "cache_hits": getattr(self, '_cache_hits', 0)
        }
    
    def clear_cache(self):
        """清理缓存"""
        self.content_cache.clear()
        print("✓ embedding缓存已清理")
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个嵌入向量的余弦相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            
        Returns:
            相似度分数 (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity) 