"""
知识图谱向量存储模块
在Milvus中存储实体向量，支持语义搜索和实体匹配
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Optional, Tuple
import json
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

from config.config import config
from .knowledge_graph import Entity, EntityType
from .embeddings import EmbeddingGenerator

class KGVectorStore:
    """知识图谱向量存储器"""
    
    def __init__(self):
        """初始化向量存储器"""
        print("初始化知识图谱向量存储:")
        print(f"  服务器: {config.milvus_host}:{config.milvus_port}")
        print(f"  集合名称: {config.kg_collection_name}")
        print(f"  向量维度: {config.kg_vector_dimension}")
        
        self.collection_name = config.kg_collection_name
        self.vector_dimension = config.kg_vector_dimension
        self.collection = None
        self.embedding_generator = EmbeddingGenerator()
        
        self._connect()
    
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
        """检查集合是否存在"""
        try:
            return utility.has_collection(self.collection_name)
        except Exception as e:
            print(f"⚠️ 检查集合存在性失败: {e}")
            return False
    
    def create_collection(self):
        """创建知识图谱向量集合"""
        try:
            if utility.has_collection(self.collection_name):
                print(f"✓ 集合 '{self.collection_name}' 已存在")
                self.collection = Collection(self.collection_name)
                return
            
            # 定义字段schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=500),
                FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="properties", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="source_papers", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="confidence", dtype=DataType.FLOAT),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dimension),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=100)
            ]
            
            # 创建集合schema
            schema = CollectionSchema(fields, "Photovoltaic Knowledge Graph Entities")
            
            # 创建集合
            self.collection = Collection(self.collection_name, schema)
            print(f"✓ 成功创建知识图谱集合 '{self.collection_name}'")
            
        except Exception as e:
            print(f"✗ 创建集合失败: {e}")
            raise
    
    def create_index(self):
        """创建向量索引"""
        try:
            if self.collection.has_index():
                print("✓ 向量索引已存在")
                return
            
            # 创建向量索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index("embedding", index_params)
            print("✓ 成功创建知识图谱向量索引")
            
        except Exception as e:
            print(f"✗ 创建索引失败: {e}")
            raise
    
    def load_collection(self):
        """加载集合到内存"""
        try:
            self.collection.load()
            print(f"✓ 集合 '{self.collection_name}' 已加载")
        except Exception as e:
            print(f"✗ 加载集合失败: {e}")
            raise
    
    def insert_entity(self, entity: Entity) -> bool:
        """插入单个实体向量"""
        try:
            # 生成实体描述文本用于向量化
            description_text = self._generate_entity_text(entity)
            
            # 生成向量
            embedding = self.embedding_generator.generate_embedding(description_text)
            
            # 准备数据
            data = [
                [f"kg_{entity.id}"],  # id
                [entity.id],          # entity_id
                [entity.name],        # entity_name
                [entity.type.value],  # entity_type
                [entity.description], # description
                [json.dumps(entity.properties)],  # properties
                [json.dumps(entity.source_papers)],  # source_papers
                [entity.confidence],  # confidence
                [embedding],          # embedding
                [entity.created_at]   # created_at
            ]
            
            # 插入数据
            self.collection.insert(data)
            self.collection.flush()
            
            return True
            
        except Exception as e:
            print(f"✗ 插入实体向量失败: {e}")
            return False
    
    def batch_insert_entities(self, entities: List[Entity]) -> int:
        """批量插入实体向量"""
        if not entities:
            return 0
        
        print(f"批量插入 {len(entities)} 个实体向量...")
        
        try:
            # 准备批量数据
            ids = []
            entity_ids = []
            entity_names = []
            entity_types = []
            descriptions = []
            properties_list = []
            source_papers_list = []
            confidences = []
            embeddings = []
            created_ats = []
            
            # 生成所有实体的描述文本
            description_texts = []
            for entity in entities:
                description_text = self._generate_entity_text(entity)
                description_texts.append(description_text)
            
            # 批量生成向量
            print("  生成实体向量...")
            batch_embeddings = self.embedding_generator.generate_embeddings_batch(description_texts)
            
            # 准备插入数据
            for i, entity in enumerate(entities):
                if i < len(batch_embeddings):
                    ids.append(f"kg_{entity.id}")
                    entity_ids.append(entity.id)
                    entity_names.append(entity.name)
                    entity_types.append(entity.type.value)
                    descriptions.append(entity.description)
                    properties_list.append(json.dumps(entity.properties))
                    source_papers_list.append(json.dumps(entity.source_papers))
                    confidences.append(entity.confidence)
                    embeddings.append(batch_embeddings[i])
                    created_ats.append(entity.created_at)
            
            # 批量插入
            data = [
                ids, entity_ids, entity_names, entity_types, descriptions,
                properties_list, source_papers_list, confidences, embeddings, created_ats
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
            print(f"✓ 成功插入 {len(ids)} 个实体向量")
            return len(ids)
            
        except Exception as e:
            print(f"✗ 批量插入失败: {e}")
            return 0
    
    def _generate_entity_text(self, entity: Entity) -> str:
        """生成实体的文本描述用于向量化"""
        text_parts = [
            f"Entity: {entity.name}",
            f"Type: {entity.type.value}",
        ]
        
        if entity.description:
            text_parts.append(f"Description: {entity.description}")
        
        # 添加重要属性
        if entity.properties:
            properties_text = []
            for key, value in entity.properties.items():
                if isinstance(value, (str, int, float)):
                    properties_text.append(f"{key}: {value}")
            if properties_text:
                text_parts.append("Properties: " + ", ".join(properties_text))
        
        return " | ".join(text_parts)
    
    def search_similar_entities(self, query_text: str, entity_type: str = None, 
                               top_k: int = 10, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """搜索相似实体"""
        try:
            # Ensure collection is loaded
            if self.collection is None:
                self.create_collection()
                self.create_index()
            
            self.load_collection()
            
            # 生成查询向量
            query_embedding = self.embedding_generator.generate_embedding(query_text, is_query=True)
            
            # 构建搜索表达式
            if entity_type:
                expr = f'entity_type == "{entity_type}"'
            else:
                expr = None
            
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
                expr=expr,
                output_fields=[
                    "entity_id", "entity_name", "entity_type", 
                    "description", "properties", "confidence"
                ]
            )
            
            # 处理结果
            entities = []
            for hits in results:
                for hit in hits:
                    # Lower threshold and add debug info
                    print(f"   Debug: Found entity {hit.entity.get('entity_name', 'Unknown')} with score {hit.score:.3f}")
                    if hit.score >= score_threshold:
                        entity_data = {
                            "entity_id": hit.entity.get("entity_id"),
                            "entity_name": hit.entity.get("entity_name"),  
                            "entity_type": hit.entity.get("entity_type"),
                            "description": hit.entity.get("description"),
                            "properties": json.loads(hit.entity.get("properties", "{}")),
                            "confidence": hit.entity.get("confidence"),
                            "similarity_score": hit.score
                        }
                        entities.append(entity_data)
            
            return entities
            
        except Exception as e:
            print(f"✗ 搜索相似实体失败: {e}")
            return []
    
    def find_entity_by_name(self, entity_name: str, entity_type: str = None) -> Optional[Dict[str, Any]]:
        """根据名称查找实体"""
        try:
            # 构建查询表达式
            expr = f'entity_name == "{entity_name}"'
            if entity_type:
                expr += f' && entity_type == "{entity_type}"'
            
            # 执行查询
            results = self.collection.query(
                expr=expr,
                output_fields=[
                    "entity_id", "entity_name", "entity_type", 
                    "description", "properties", "confidence"
                ],
                limit=1
            )
            
            if results:
                result = results[0]
                return {
                    "entity_id": result.get("entity_id"),
                    "entity_name": result.get("entity_name"),
                    "entity_type": result.get("entity_type"),
                    "description": result.get("description"),
                    "properties": json.loads(result.get("properties", "{}")),
                    "confidence": result.get("confidence")
                }
            
            return None
            
        except Exception as e:
            print(f"✗ 查找实体失败: {e}")
            return None
    
    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取指定类型的所有实体"""
        try:
            expr = f'entity_type == "{entity_type}"'
            
            results = self.collection.query(
                expr=expr,
                output_fields=[
                    "entity_id", "entity_name", "entity_type", 
                    "description", "properties", "confidence"
                ],
                limit=limit
            )
            
            entities = []
            for result in results:
                entity_data = {
                    "entity_id": result.get("entity_id"),
                    "entity_name": result.get("entity_name"),
                    "entity_type": result.get("entity_type"),
                    "description": result.get("description"),
                    "properties": json.loads(result.get("properties", "{}")),
                    "confidence": result.get("confidence")
                }
                entities.append(entity_data)
            
            return entities
            
        except Exception as e:
            print(f"✗ 获取实体失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        try:
            # 获取总数
            total_entities = self.collection.num_entities
            
            # 获取各类型实体数量
            entity_types = {}
            for entity_type in EntityType:
                expr = f'entity_type == "{entity_type.value}"'
                results = self.collection.query(
                    expr=expr,
                    output_fields=["entity_id"],
                    limit=10000  # 估算用，实际应该用聚合查询
                )
                entity_types[entity_type.value] = len(results)
            
            return {
                "total_entities": total_entities,
                "entity_types": entity_types,
                "collection_name": self.collection_name,
                "vector_dimension": self.vector_dimension
            }
            
        except Exception as e:
            print(f"✗ 获取统计信息失败: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"✓ 已清空集合 '{self.collection_name}'")
                return True
            return True
        except Exception as e:
            print(f"✗ 清空集合失败: {e}")
            return False 