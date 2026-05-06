"""
向量存储模块
使用Milvus进行向量存储和检索
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from config.config import config

class MilvusVectorStore:
    """Milvus向量存储器"""
    
    def __init__(self):
        """初始化向量存储器"""
        print("初始化Milvus向量存储:")
        print(f"  服务器: {config.milvus_host}:{config.milvus_port}")
        print(f"  集合名称: {config.collection_name}")
        print(f"  向量维度: {config.vector_dimension}")
        
        self.collection_name = config.collection_name
        self.vector_dimension = config.vector_dimension
        self.collection = None
        self._connect()
        
        # 尝试加载已存在的collection
        if self.collection_exists():
            self.collection = Collection(self.collection_name)
            # 必须加载到内存才能搜索
            try:
                self.collection.load()
                print(f"✓ 已加载集合 '{self.collection_name}' 到内存")
            except Exception as e:
                print(f"⚠️  加载集合到内存失败: {e}")
                # 尝试通过 load_collection 方法
                self.load_collection()
    
    def _connect(self):
        """连接到Milvus服务器"""
        try:
            connections.connect(
                alias="default", 
                host=config.milvus_host, 
                port=config.milvus_port
            )
            print("✓ 成功连接到Milvus服务器")
        except Exception as e:
            print(f"✗ 连接Milvus失败: {e}")
            raise
    
    def collection_exists(self) -> bool:
        """
        检查集合是否存在
        
        Returns:
            集合是否存在
        """
        try:
            return utility.has_collection(self.collection_name)
        except Exception as e:
            print(f"⚠️ 检查集合存在性失败: {e}")
            return False
    
    def create_collection(self):
        """创建集合"""
        try:
            if utility.has_collection(self.collection_name):
                print(f"✓ 集合 '{self.collection_name}' 已存在")
                self.collection = Collection(self.collection_name)
                return
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)
            ]
            
            # 创建集合schema
            schema = CollectionSchema(fields, "Academic papers collection")
            
            # 创建集合
            self.collection = Collection(self.collection_name, schema)
            print(f"✓ 成功创建集合 '{self.collection_name}'")
            
        except Exception as e:
            print(f"✗ 创建集合失败: {e}")
            raise
    
    def create_index(self):
        """创建向量索引"""
        try:
            if self.collection.has_index():
                print("✓ 向量索引已存在")
                return
            
            # 定义索引参数
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            # 创建索引
            self.collection.create_index("embedding", index_params)
            print("✓ 成功创建向量索引")
            
        except Exception as e:
            print(f"✗ 创建索引失败: {e}")
            raise
    
    def load_collection(self):
        """加载集合到内存"""
        try:
            # 如果集合不存在，先获取它
            if self.collection is None:
                if utility.has_collection(self.collection_name):
                    self.collection = Collection(self.collection_name)
                else:
                    print(f"⚠️  集合 '{self.collection_name}' 不存在")
                    return False
            
            self.collection.load()
            print(f"✓ 集合 '{self.collection_name}' 已加载")
            return True
        except Exception as e:
            print(f"✗ 加载集合失败: {e}")
            return False
    
    def insert_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        插入文档向量
        
        Args:
            chunks: 文档分块列表
            embeddings: 对应的嵌入向量列表
        """
        if len(chunks) != len(embeddings):
            raise ValueError("文档数量与嵌入向量数量不匹配")
        
        try:
            # 准备数据 - 使用列表格式按字段顺序组织
            embeddings_data = []
            content_data = []
            source_data = []
            chunk_id_data = []
            metadata_data = []
            
            for chunk, embedding in zip(chunks, embeddings):
                metadata = chunk.get('metadata', {})
                
                embeddings_data.append(embedding)
                content_data.append(chunk['content'][:65535])  # 限制长度
                source_data.append(metadata.get('source', '')[:1000])  # 限制长度
                chunk_id_data.append(metadata.get('chunk_id', 0))
                metadata_data.append(str(metadata)[:2000])  # 转换为字符串并限制长度
            
            # 按字段组织数据
            data = [
                embeddings_data,
                content_data,
                source_data,
                chunk_id_data,
                metadata_data
            ]
            
                    # 插入数据
            self.collection.insert(data)
            self.collection.flush()
            
            print(f"✓ 成功插入 {len(embeddings_data)} 个文档向量")
            
        except Exception as e:
            print(f"✗ 插入向量失败: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 检查 collection 是否已初始化
            if self.collection is None:
                print(f"⚠️  Collection 未初始化，尝试重新加载...")
                if self.collection_exists():
                    self.collection = Collection(self.collection_name)
                    self.collection.load()
                else:
                    print(f"✗ Collection '{self.collection_name}' 不存在")
                    return []
            
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "source", "chunk_id", "metadata"]
            )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        'id': hit.id,
                        'score': float(hit.score),
                        'content': hit.entity.get('content'),
                        'source': hit.entity.get('source'),
                        'chunk_id': hit.entity.get('chunk_id'),
                        'metadata': hit.entity.get('metadata')
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"✗ 搜索向量失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            stats = {
                'collection_name': self.collection_name,
                'vector_dimension': self.vector_dimension,
                'total_entities': 0,
                'has_index': False,
                'is_loaded': False
            }
            
            if self.collection:
                stats['total_entities'] = self.collection.num_entities
                stats['has_index'] = self.collection.has_index()
                # 检查是否已加载（这个方法可能在不同版本的pymilvus中有所不同）
                try:
                    self.collection.query(expr="id >= 0", limit=1)
                    stats['is_loaded'] = True
                except:
                    stats['is_loaded'] = False
            
            return stats
            
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {'error': str(e)}
    
    def delete_collection(self):
        """删除集合"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"✓ 已删除集合 '{self.collection_name}'")
            else:
                print(f"集合 '{self.collection_name}' 不存在")
        except Exception as e:
            print(f"✗ 删除集合失败: {e}")
            raise
    
    def drop_collection(self):
        """删除集合（别名方法）"""
        self.delete_collection()
    
    def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect("default")
            print("✓ 已断开Milvus连接")
        except Exception as e:
            print(f"断开连接时发生错误: {e}") 