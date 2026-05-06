"""
光电文献知识图谱模块
结合Neo4j图数据库和Milvus向量存储，构建领域知识图谱
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class EntityType(Enum):
    """实体类型枚举"""
    MATERIAL = "Material"           # 材料：钙钛矿、硅、有机材料等
    DEVICE = "Device"             # 器件：太阳能电池、LED、激光器等
    METRIC = "Metric"             # 性能指标：效率、带隙、迁移率等
    PROCESS = "Process"           # 制备工艺：旋涂、蒸镀、退火等
    AUTHOR = "Author"             # 作者
    INSTITUTION = "Institution"   # 机构
    PAPER = "Paper"               # 论文
    CONCEPT = "Concept"           # 概念：能级、载流子等
    MEASUREMENT = "Measurement"   # 测量方法：CV、EQE、PL等
    PARAMETER = "Parameter"      # 参数：温度、光照强度、湿度等
    FORMULA = "Formula"           # 公式：数学方程、表达式等

class RelationType(Enum):
    """关系类型枚举 - v3.1 增强版（13种关系类型）"""
    # 原有关系类型（10种）
    USES = "USES"                     # 使用关系：器件使用材料
    HAS_PROPERTY = "HAS_PROPERTY"     # 属性关系：材料具有性能
    FABRICATED_BY = "FABRICATED_BY"   # 制备关系：材料通过工艺制备
    AUTHORED_BY = "AUTHORED_BY"       # 作者关系：论文由作者发表
    AFFILIATED_WITH = "AFFILIATED_WITH" # 隶属关系：作者隶属机构
    MEASURES = "MEASURES"             # 测量关系：方法测量指标
    IMPROVES = "IMPROVES"             # 改进关系：工艺改进性能
    CITES = "CITES"                  # 引用关系：论文引用论文
    CONTAINS = "CONTAINS"             # 包含关系：论文包含概念
    RELATED_TO = "RELATED_TO"         # 相关关系：通用关联
    INFLUENCES = "INFLUENCES"         # 影响关系：参数影响指标
    IS_A = "IS_A"                     # 分类关系：PM6是一种聚合物
    # v3.0 新增关系类型（3种）
    PREDICTS = "PREDICTS"             # 预测关系：ML算法预测性能
    OPTIMIZES = "OPTIMIZES"           # 优化关系：方法优化性能
    TRAINED_ON = "TRAINED_ON"         # 训练关系：模型在数据上训练
    # v3.1 新增关系类型（3种）
    COMPOSES = "COMPOSES"             # 组成关系：实体由其他实体组成
    ENHANCES = "ENHANCES"             # 增强关系：实体增强另一实体的性质
    CORRELATES_WITH = "CORRELATES_WITH" # 相关关系：两个实体之间存在科学相关性

@dataclass
class Entity:
    """知识图谱实体"""
    id: str                    # 唯一标识（基于名称+类型生成）
    name: str                  # 实体名称
    type: EntityType           # 实体类型
    properties: Dict[str, Any] # 属性字典
    description: str = ""      # 主要描述
    descriptions: List[str] = None  # 多个描述列表（支持来自不同来源的描述）
    source_papers: List[str] = None  # 来源论文ID列表
    confidence: float = 1.0    # 置信度
    created_at: str = None
    
    def __post_init__(self):
        if self.source_papers is None:
            self.source_papers = []
        if self.descriptions is None:
            self.descriptions = []
            # 如果有主要描述，添加到描述列表中
            if self.description:
                self.descriptions.append(self.description)
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    @staticmethod
    def generate_id(name: str, entity_type: EntityType) -> str:
        """
        生成基于名称和类型的唯一ID
        支持缩写和全称识别（如PCE和Power Conversion Efficiency生成相同ID）
        """
        import hashlib
        from .entity_normalizer import get_normalizer
        
        # 使用标准化器获取标准名称（识别缩写和全称）
        normalizer = get_normalizer()
        canonical_name = normalizer.get_canonical_name(name, entity_type.value)
        
        # 使用标准名称和类型生成唯一ID
        unique_string = f"{canonical_name}_{entity_type.value}"
        # 使用hash确保ID不会太长
        hash_value = hashlib.md5(unique_string.encode()).hexdigest()[:12]
        return f"{entity_type.value}_{hash_value}"

@dataclass 
class Relation:
    """知识图谱关系"""
    id: str                    # 唯一标识
    source_id: str             # 源实体ID
    target_id: str             # 目标实体ID
    type: RelationType         # 关系类型
    properties: Dict[str, Any] # 关系属性
    description: str = ""      # 描述
    source_papers: List[str] = None  # 来源论文ID列表
    confidence: float = 1.0    # 置信度
    created_at: str = None
    
    def __post_init__(self):
        if self.source_papers is None:
            self.source_papers = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class PhotovoltaicKnowledgeGraph:
    """光电知识图谱管理器"""
    
    def __init__(self, use_traceable_extractor: bool = True, enable_traceability: bool = True, use_enhanced_extractor: bool = True):
        """
        初始化知识图谱
        
        Args:
            use_traceable_extractor: 是否使用新的可追溯光电提取器（推荐）
            enable_traceability: 是否启用追溯功能（仅当use_traceable_extractor=True时有效）
            use_enhanced_extractor: 是否使用增强版提取器（最新版本，推荐）
        """
        from config.config import config
        
        self.config = config
        self.use_traceable_extractor = use_traceable_extractor
        self.use_enhanced_extractor = use_enhanced_extractor
        
        # 初始化LLM抽取器
        if use_traceable_extractor:
            # Try to use enhanced extractor first
            if use_enhanced_extractor:
                try:
                    from .optoelectronic_kg_extractor_enhanced import EnhancedOptoelectronicKGExtractor
                    self.llm_extractor = EnhancedOptoelectronicKGExtractor(enable_traceability=enable_traceability)
                    print(f"🧠 Initializing Photovoltaic Knowledge Graph System (Enhanced Optoelectronic Extractor)")
                except ImportError as e:
                    print(f"⚠️ Failed to import EnhancedOptoelectronicKGExtractor: {e}")
                    print(f"⚠️ Falling back to standard traceable extractor")
                    try:
                        from .optoelectronic_kg_extractor import OptoelectronicKGExtractor
                        self.llm_extractor = OptoelectronicKGExtractor(enable_traceability=enable_traceability)
                        print(f"🧠 Initializing Photovoltaic Knowledge Graph System (Traceable Optoelectronic Extractor)")
                    except ImportError as e2:
                        print(f"⚠️ Failed to import OptoelectronicKGExtractor, falling back to legacy extractor: {e2}")
                        from .llm_extractor import LLMEntityRelationExtractor
                        self.llm_extractor = LLMEntityRelationExtractor()
                        self.use_traceable_extractor = False
                        self.use_enhanced_extractor = False
                        print("🧠 Initializing Photovoltaic Knowledge Graph System (Legacy Extractor)")
            else:
                # Use standard traceable extractor
                try:
                    from .optoelectronic_kg_extractor import OptoelectronicKGExtractor
                    self.llm_extractor = OptoelectronicKGExtractor(enable_traceability=enable_traceability)
                    print(f"🧠 Initializing Photovoltaic Knowledge Graph System (Traceable Optoelectronic Extractor)")
                except ImportError as e:
                    print(f"⚠️ Failed to import OptoelectronicKGExtractor, falling back to legacy extractor: {e}")
                    from .llm_extractor import LLMEntityRelationExtractor
                    self.llm_extractor = LLMEntityRelationExtractor()
                    self.use_traceable_extractor = False
                    print("🧠 Initializing Photovoltaic Knowledge Graph System (Legacy Extractor)")
        else:
            from .llm_extractor import LLMEntityRelationExtractor
            self.llm_extractor = LLMEntityRelationExtractor()
            print("🧠 Initializing Photovoltaic Knowledge Graph System (Legacy Extractor)")
        
        # 存储抽取的实体和关系
        self.extracted_entities = []
        self.extracted_relations = []
    
    def get_extracted_entities(self) -> List[Entity]:
        """获取抽取的实体"""
        return self.extracted_entities
    
    def get_extracted_relations(self) -> List[Relation]:
        """获取抽取的关系"""
        return self.extracted_relations
    
    def build_kg_from_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """从文档集合构建知识图谱（使用智能分块处理长文档）"""
        print("🔨 Starting to build photovoltaic literature knowledge graph...")
        
        # 导入智能分块器
        from core import PhotovoltaicTextSplitter
        
        # 初始化智能分块器
        splitter = PhotovoltaicTextSplitter()
        
        # 准备所有chunks
        all_text_chunks = []
        all_paper_ids = []
        all_metadata_list = []
        
        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            if len(content.strip()) < 50:  # Skip too short documents
                continue
                
            paper_id = doc.get('metadata', {}).get('source', f'paper_{i}')
            doc_metadata = doc.get('metadata', {})
            
            # 使用智能分块器处理文档（移除3000字符限制）
            print(f"  📄 Processing document {i+1}/{len(documents)}: {paper_id} ({len(content)} 字符)")
            
            # 智能分块
            chunks = splitter.split_text(content)
            print(f"     → 分为 {len(chunks)} 个智能chunks")
            
            # 为每个chunk准备元数据
            for chunk_idx, chunk in enumerate(chunks):
                chunk_content = chunk.get('content', '')
                chunk_metadata = chunk.get('metadata', {})
                
                # 合并文档元数据和chunk元数据
                combined_metadata = {
                    **doc_metadata,
                    **chunk_metadata,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
                
                all_text_chunks.append(chunk_content)
                all_paper_ids.append(paper_id)
                all_metadata_list.append(combined_metadata)
        
        print(f"  📊 Total: {len(all_text_chunks)} chunks from {len(documents)} documents")
        
        # Use LLM to batch extract entities and relations
        if self.use_traceable_extractor:
            # Use new traceable extractor with metadata support
            all_entities, all_relations = self.llm_extractor.extract_entities_and_relations_batch(
                text_chunks=all_text_chunks,
                paper_id="batch_extraction",
                metadata_list=all_metadata_list
            )
        else:
            # Use legacy extractor (also supports chunking now)
            all_entities, all_relations = self.llm_extractor.extract_entities_and_relations_batch(
                text_chunks=all_text_chunks,
                paper_ids=all_paper_ids
            )
        
        # Deduplicate and merge
        unique_entities = self._deduplicate_entities(all_entities)
        unique_relations = self._deduplicate_relations(all_relations)
        
        # Store results for later use
        self.extracted_entities = unique_entities
        self.extracted_relations = unique_relations
        
        print(f"✓ LLM extraction completed: {len(unique_entities)} entities, {len(unique_relations)} relations")
        
        # Print traceability statistics if using traceable extractor
        if self.use_traceable_extractor and hasattr(self.llm_extractor, 'print_statistics'):
            self.llm_extractor.print_statistics()
        
        return {
            "entities": len(unique_entities),
            "relations": len(unique_relations),
            "documents": len(documents)  # 使用原始documents列表的长度
        }
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        实体去重和合并
        - 基于实体名称（标准化，支持缩写识别）和类型进行合并
        - 保留所有不同的描述
        - 合并属性和来源论文
        - 识别PCE和Power Conversion Efficiency为同一实体
        """
        from .entity_normalizer import get_normalizer
        
        normalizer = get_normalizer()
        entity_map = {}
        
        for entity in entities:
            # 使用标准化器生成唯一键（识别缩写和全称）
            canonical_name = normalizer.get_canonical_name(entity.name, entity.type.value)
            unique_key = f"{canonical_name}_{entity.type.value}"
            
            if unique_key in entity_map:
                # 实体已存在，进行合并
                existing = entity_map[unique_key]
                
                # 1. 合并源论文列表
                existing.source_papers.extend(entity.source_papers)
                existing.source_papers = list(set(existing.source_papers))
                
                # 2. 合并描述列表（去重，保留所有不同的描述）
                if entity.description and entity.description.strip():
                    if entity.description not in existing.descriptions:
                        existing.descriptions.append(entity.description)
                # 合并新实体的描述列表
                for desc in entity.descriptions:
                    if desc and desc.strip() and desc not in existing.descriptions:
                        existing.descriptions.append(desc)
                
                # 3. 合并属性（新属性添加，相同key的值合并为列表）
                for key, value in entity.properties.items():
                    if key in existing.properties:
                        # 如果属性已存在，将值合并为列表
                        existing_value = existing.properties[key]
                        if isinstance(existing_value, list):
                            if value not in existing_value:
                                existing_value.append(value)
                        else:
                            if existing_value != value:
                                existing.properties[key] = [existing_value, value]
                    else:
                        existing.properties[key] = value
                
                # 4. 更新置信度（取最高值）
                existing.confidence = max(existing.confidence, entity.confidence)
                
                # 5. 更新主要描述为最长的描述
                if existing.descriptions:
                    existing.description = max(existing.descriptions, key=len)
                
                # 6. 更新实体名称为最佳名称（优先全称）
                # 收集所有名称变体
                all_names = [existing.name, entity.name]
                best_name = normalizer.merge_entity_names(all_names, existing.type.value)
                existing.name = best_name
            else:
                # 新实体，更新ID为标准化ID
                entity.id = Entity.generate_id(entity.name, entity.type)
                # 标准化实体名称（优先使用全称）
                entity.name = normalizer.normalize_entity_name(entity.name, entity.type.value)
                entity_map[unique_key] = entity
        
        return list(entity_map.values())
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """
        关系去重和合并
        - 基于源实体ID、目标实体ID和关系类型进行合并
        - 识别跨文献的相同关系
        - 合并描述、属性和来源论文
        """
        relation_map = {}
        
        for relation in relations:
            # 生成基于实体和关系类型的唯一键
            unique_key = f"{relation.source_id}_{relation.type.value}_{relation.target_id}"
            
            if unique_key in relation_map:
                # 关系已存在，进行合并
                existing = relation_map[unique_key]
                
                # 1. 合并源论文列表
                existing.source_papers.extend(relation.source_papers)
                existing.source_papers = list(set(existing.source_papers))
                
                # 2. 合并描述（如果不同）
                if relation.description and relation.description != existing.description:
                    if existing.description:
                        existing.description += f"; {relation.description}"
                    else:
                        existing.description = relation.description
                
                # 3. 合并属性（智能合并）
                for key, value in relation.properties.items():
                    if key in existing.properties:
                        existing_value = existing.properties[key]
                        if isinstance(existing_value, list):
                            if value not in existing_value:
                                existing_value.append(value)
                        else:
                            if existing_value != value:
                                existing.properties[key] = [existing_value, value]
                    else:
                        existing.properties[key] = value
                
                # 4. 更新置信度（取最高值）
                existing.confidence = max(existing.confidence, relation.confidence)
            else:
                # 新关系
                relation_map[unique_key] = relation
        
        return list(relation_map.values()) 