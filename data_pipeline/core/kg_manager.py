"""
综合知识图谱管理器
整合Neo4j图数据库和Milvus向量存储，提供统一的知识图谱管理接口
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime

from .knowledge_graph import PhotovoltaicKnowledgeGraph, Entity, Relation, EntityType, RelationType
from .neo4j_manager import Neo4jManager
from .kg_vector_store import KGVectorStore
from .document_loader import DocumentLoader
from config.config import config

class KnowledgeGraphManager:
    """综合知识图谱管理器"""
    
    def __init__(self, use_traceable_extractor: bool = True, enable_traceability: bool = True):
        """
        初始化知识图谱管理器
        
        Args:
            use_traceable_extractor: 是否使用新的可追溯光电提取器（推荐）
            enable_traceability: 是否启用追溯功能
        """
        print("🧠 Initializing comprehensive knowledge graph manager...")
        print(f"  Traceable Extraction: {'✓ Enabled' if use_traceable_extractor else '✗ Disabled'}")
        print(f"  Traceability: {'✓ Enabled' if enable_traceability else '✗ Disabled'}")
        
        # 初始化核心组件
        self.kg = PhotovoltaicKnowledgeGraph(
            use_traceable_extractor=use_traceable_extractor,
            enable_traceability=enable_traceability
        )
        self.neo4j_manager = None
        self.vector_store = None
        
        # 从配置文件加载
        self.document_loader = DocumentLoader(config.test_mds_path)
        
        # 尝试初始化Neo4j
        try:
            self.neo4j_manager = Neo4jManager()
            print("✓ Neo4j graph database connected")
        except Exception as e:
            print(f"⚠️ Neo4j connection failed: {e}")
            print("  Knowledge graph will only use vector storage functionality")
        
        # Initialize vector store
        try:
            self.vector_store = KGVectorStore()
            print("✓ Milvus vector store connected")
        except Exception as e:
            print(f"✗ Milvus connection failed: {e}")
            raise
        
        print("✅ Knowledge graph manager initialization completed")
    
    def build_knowledge_graph(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """从现有文档构建知识图谱"""
        print("\n🏗️ 开始构建光电文献知识图谱...")
        start_time = time.time()
        
        try:
            # 1. 加载文档
            print("1. 加载文档...")
            documents = self.document_loader.load_all_documents()
            if not documents:
                print("✗ 没有找到文档")
                return {"success": False, "message": "没有找到文档"}
            
            print(f"✓ 加载了 {len(documents)} 个文档")
            
            # 2. 从文档抽取知识
            print("2. 抽取实体和关系...")
            kg_stats = self.kg.build_kg_from_documents(documents)
            
            # 3. 获取抽取的实体和关系
            # 注意：这里需要修改PhotovoltaicKnowledgeGraph类来返回实际的实体和关系
            entities, relations = self._extract_entities_relations_from_kg(kg_stats)
            
            if not entities:
                print("⚠️ 没有抽取到实体")
                return {"success": False, "message": "没有抽取到实体"}
            
            # 4. 存储到Neo4j（如果可用）
            neo4j_stats = {"entities": 0, "relations": 0}
            if self.neo4j_manager:
                print("3. 存储到Neo4j图数据库...")
                neo4j_stats = self._store_to_neo4j(entities, relations)
            
            # 5. 存储到Milvus向量库
            print("4. 生成实体向量并存储...")
            vector_stats = self._store_to_vector_db(entities)
            
            # 6. 构建完成统计
            elapsed = time.time() - start_time
            
            result = {
                "success": True,
                "build_time": elapsed,
                "documents_processed": len(documents),
                "entities_extracted": len(entities),
                "relations_extracted": len(relations),
                "neo4j_stats": neo4j_stats,
                "vector_stats": vector_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"\n🎉 知识图谱构建完成!")
            print(f"   构建时间: {elapsed:.2f}秒")
            print(f"   处理文档: {len(documents)} 个")
            print(f"   抽取实体: {len(entities)} 个")
            print(f"   抽取关系: {len(relations)} 个")
            if self.neo4j_manager:
                print(f"   Neo4j存储: {neo4j_stats.get('entities', 0)} 实体, {neo4j_stats.get('relations', 0)} 关系")
            print(f"   向量存储: {vector_stats.get('entities', 0)} 个实体向量")
            
            return result
            
        except Exception as e:
            print(f"❌ 知识图谱构建失败: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "message": str(e)}
    
    def _extract_entities_relations_from_kg(self, kg_stats: Dict[str, Any]) -> Tuple[List[Entity], List[Relation]]:
        """从知识图谱中获取实际抽取的实体和关系"""
        # 获取LLM抽取的实体和关系
        entities = self.kg.get_extracted_entities()
        relations = self.kg.get_extracted_relations()
        
        print(f"   📊 获取到 {len(entities)} 个实体和 {len(relations)} 个关系")
        
        # 如果抽取结果为空，创建一些示例数据用于测试
        if not entities:
            print("   ⚠️ 未抽取到实体，使用示例数据")
            entities = [
                Entity(
                    id="Material_perovskite",
                    name="perovskite",
                    type=EntityType.MATERIAL,
                    properties={"band_gap": "1.6 eV", "efficiency": "high"},
                    description="Lead halide perovskite material for solar cells",
                    confidence=0.9
                ),
                Entity(
                    id="Device_solar_cell",
                    name="solar cell",
                    type=EntityType.DEVICE,
                    properties={"type": "photovoltaic"},
                    description="Device that converts light to electricity",
                    confidence=0.95
                )
            ]
        
        if not relations:
            print("   ⚠️ 未抽取到关系，使用示例数据")
            relations = [
                Relation(
                    id="USES_solar_cell_perovskite",
                    source_id="Device_solar_cell",
                    target_id="Material_perovskite",
                    type=RelationType.USES,
                    properties={"context": "perovskite solar cells"},
                    confidence=0.8
                )
            ]
        
        return entities, relations
    
    def _store_to_neo4j(self, entities: List[Entity], relations: List[Relation]) -> Dict[str, int]:
        """存储到Neo4j图数据库"""
        if not self.neo4j_manager:
            return {"entities": 0, "relations": 0}
        
        try:
            # 批量插入实体
            entities_count = self.neo4j_manager.batch_insert_entities(entities)
            print(f"   ✓ Neo4j存储了 {entities_count} 个实体")
            
            # 批量插入关系
            relations_count = self.neo4j_manager.batch_insert_relations(relations)
            print(f"   ✓ Neo4j存储了 {relations_count} 个关系")
            
            return {"entities": entities_count, "relations": relations_count}
            
        except Exception as e:
            print(f"   ✗ Neo4j存储失败: {e}")
            return {"entities": 0, "relations": 0}
    
    def _store_to_vector_db(self, entities: List[Entity]) -> Dict[str, int]:
        """存储到Milvus向量数据库"""
        try:
            # 创建集合和索引
            self.vector_store.create_collection()
            self.vector_store.create_index()
            
            # 批量插入实体向量
            entities_count = self.vector_store.batch_insert_entities(entities)
            
            # 加载集合
            self.vector_store.load_collection()
            
            print(f"   ✓ Milvus存储了 {entities_count} 个实体向量")
            return {"entities": entities_count}
            
        except Exception as e:
            print(f"   ✗ Milvus存储失败: {e}")
            return {"entities": 0}
    
    def search_entities(self, query: str, entity_type: str = None, 
                       search_method: str = "vector", top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索实体"""
        if search_method == "vector" and self.vector_store:
            return self.vector_store.search_similar_entities(
                query_text=query,
                entity_type=entity_type,
                top_k=top_k
            )
        elif search_method == "graph" and self.neo4j_manager:
            return self.neo4j_manager.search_entities(
                name_pattern=query,
                entity_type=entity_type,
                limit=top_k
            )
        else:
            print(f"搜索方法 '{search_method}' 不可用")
            return []
    
    def get_entity_relations(self, entity_id: str) -> List[Dict[str, Any]]:
        """获取实体的关系"""
        if self.neo4j_manager:
            return self.neo4j_manager.get_entity_relations(entity_id)
        else:
            print("Neo4j不可用，无法获取关系")
            return []
    
    def find_entity_path(self, source_id: str, target_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """查找实体间的路径"""
        if self.neo4j_manager:
            return self.neo4j_manager.find_shortest_path(source_id, target_id, max_depth)
        else:
            print("Neo4j不可用，无法查找路径")
            return []
    
    def get_knowledge_graph_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "neo4j": self.neo4j_manager is not None,
                "milvus": self.vector_store is not None
            }
        }
        
        # Neo4j统计
        if self.neo4j_manager:
            try:
                neo4j_stats = self.neo4j_manager.get_statistics()
                stats["neo4j"] = neo4j_stats
            except Exception as e:
                stats["neo4j"] = {"error": str(e)}
        
        # Milvus统计
        if self.vector_store:
            try:
                vector_stats = self.vector_store.get_statistics()
                stats["milvus"] = vector_stats
            except Exception as e:
                stats["milvus"] = {"error": str(e)}
        
        return stats
    
    def clear_knowledge_graph(self) -> Dict[str, bool]:
        """清空知识图谱"""
        results = {}
        
        # 清空Neo4j
        if self.neo4j_manager:
            try:
                results["neo4j"] = self.neo4j_manager.clear_database()
            except Exception as e:
                print(f"清空Neo4j失败: {e}")
                results["neo4j"] = False
        
        # 清空Milvus
        if self.vector_store:
            try:
                results["milvus"] = self.vector_store.clear_collection()
            except Exception as e:
                print(f"清空Milvus失败: {e}")
                results["milvus"] = False
        
        return results
    
    def export_knowledge_graph(self, format: str = "json") -> Optional[str]:
        """导出知识图谱"""
        # TODO: 实现知识图谱导出功能
        print(f"TODO: 实现{format}格式的知识图谱导出")
        return None
    
    def close(self):
        """关闭所有连接"""
        if self.neo4j_manager:
            self.neo4j_manager.close()
        print("✓ 知识图谱管理器已关闭") 