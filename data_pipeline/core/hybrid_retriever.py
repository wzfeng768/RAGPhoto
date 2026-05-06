"""
混合检索器 - 结合向量检索和知识图谱查询
Hybrid Retriever combining Vector Search and Knowledge Graph
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional, Tuple
import re
from openai import OpenAI

from config.config import config
from .vector_store import MilvusVectorStore
from .neo4j_manager import Neo4jManager
from .embeddings import EmbeddingGenerator


class HybridRetriever:
    """混合检索器：向量检索 + 知识图谱增强"""
    
    def __init__(self, verbose: bool = False):
        """初始化混合检索器"""
        self.verbose = verbose
        
        # 初始化向量检索组件
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = MilvusVectorStore()
        
        # 加载集合
        self.vector_available = False
        if self.vector_store.collection_exists():
            if self.vector_store.load_collection():
                self.vector_available = True
                if self.verbose:
                    print(f"✓ Vector store ready, collection: {config.collection_name}")
        
        if not self.vector_available and self.verbose:
            print(f"⚠️  Vector store not available. Please run: python run_pipeline.py --vector")
        
        # 初始化知识图谱组件
        try:
            self.neo4j_manager = Neo4jManager()
            self.kg_available = True
            if self.verbose:
                print("✓ Knowledge graph retrieval enabled")
        except Exception as e:
            self.kg_available = False
            if self.verbose:
                print(f"⚠️  Knowledge graph unavailable: {e}")
        
        # 初始化LLM用于实体识别
        self.llm_client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract key entities from query"""
        if not self.kg_available:
            return []
        
        # Use LLM to extract entities (in English)
        prompt = f"""Extract key entities (materials, processes, metrics, etc.) from the following photovoltaic domain question.
Return ONLY a comma-separated list of entity names in English, no other explanation.

Question: {query}

Entity list:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            entities_text = response.choices[0].message.content.strip()
            # Parse entity list
            entities = [e.strip() for e in entities_text.split(',') if e.strip()]
            
            if self.verbose and entities:
                print(f"🎯 Identified entities: {entities}")
            
            return entities
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Entity extraction failed: {e}")
            return []
    
    def query_knowledge_graph(self, entities: List[str]) -> List[Dict[str, Any]]:
        """从知识图谱查询相关信息（改进版：更智能的实体匹配和更丰富的信息）"""
        if not self.kg_available or not entities:
            return []
        
        kg_results = []
        seen_entities = set()  # 避免重复
        
        for entity in entities[:5]:  # 增加到5个实体
            try:
                # 改进的查询：使用参数化查询避免注入，更智能的匹配
                # 1. 精确匹配
                # 2. 包含匹配（双向）
                # 3. 相似度匹配（使用toLower）
                entity_clean = entity.strip().lower()
                entity_query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) = $entity_lower
                   OR toLower(e.name) CONTAINS $entity_lower
                   OR $entity_lower CONTAINS toLower(e.name)
                   OR toLower(e.description) CONTAINS $entity_lower
                OPTIONAL MATCH (e)-[r:RELATES]->(target:Entity)
                OPTIONAL MATCH (source:Entity)-[r2:RELATES]->(e)
                WITH e, 
                     collect(DISTINCT {relation: r.type, target: target.name, target_type: target.type, direction: 'outgoing'}) as out_relations,
                     collect(DISTINCT {relation: r2.type, target: source.name, target_type: source.type, direction: 'incoming'}) as in_relations
                RETURN e.name as entity, 
                       e.type as type,
                       e.description as description,
                       e.properties as properties,
                       out_relations + in_relations as all_relations
                LIMIT 10
                """
                
                results = self.neo4j_manager.execute_custom_query(
                    entity_query, 
                    parameters={'entity_lower': entity_clean}
                )
                
                for result in results:
                    entity_name = result.get('entity')
                    if entity_name in seen_entities:
                        continue
                    seen_entities.add(entity_name)
                    
                    # 合并所有关系（包括出边和入边）
                    all_relations = result.get('all_relations', [])
                    relations = [r for r in all_relations if r.get('target')]
                    
                    kg_info = {
                        'entity': entity_name,
                        'type': result.get('type'),
                        'description': result.get('description', ''),
                        'properties': result.get('properties', {}),
                        'relations': relations
                    }
                    
                    # ✅ 即使没有关系，也包含实体信息（可能对回答问题有帮助）
                    kg_results.append(kg_info)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  图谱查询失败 ({entity}): {e}")
                continue
        
        if self.verbose:
            if kg_results:
                total_relations = sum(len(item.get('relations', [])) for item in kg_results)
                print(f"📊 Retrieved {len(kg_results)} entities from KG with {total_relations} relations")
            else:
                print(f"⚠️  未找到匹配的知识图谱实体")
        
        return kg_results
    
    def format_kg_knowledge(self, kg_results: List[Dict[str, Any]]) -> str:
        """Format knowledge graph results as text (改进版：包含更多信息)"""
        if not kg_results:
            return ""
        
        formatted_lines = ["=== Structured Knowledge from Knowledge Graph ===\n"]
        
        for item in kg_results:
            entity_name = item.get('entity', 'Unknown')
            entity_type = item.get('type', 'Unknown')
            description = item.get('description', '')
            properties = item.get('properties', {})
            relations = item.get('relations', [])
            
            # 实体基本信息
            formatted_lines.append(f"• {entity_name} ({entity_type}):")
            
            # 添加描述（如果有）
            if description:
                formatted_lines.append(f"  Description: {description}")
            
            # 添加属性（如果有）
            if properties:
                if isinstance(properties, str):
                    try:
                        import json
                        properties = json.loads(properties)
                    except:
                        pass
                if isinstance(properties, dict) and properties:
                    prop_str = ", ".join([f"{k}: {v}" for k, v in properties.items() if v])
                    if prop_str:
                        formatted_lines.append(f"  Properties: {prop_str}")
            
            # 添加关系（如果有）
            if relations:
                formatted_lines.append(f"  Relationships:")
                for rel in relations[:10]:  # 增加到10个关系
                    rel_type = rel.get('relation', 'RELATED')
                    target = rel.get('target', 'Unknown')
                    target_type = rel.get('target_type', '')
                    direction = rel.get('direction', '')
                    if direction == 'incoming':
                        formatted_lines.append(f"    ← {rel_type}: {target} ({target_type})")
                    else:
                        formatted_lines.append(f"    → {rel_type}: {target} ({target_type})")
            else:
                formatted_lines.append(f"  (No relationships found in knowledge graph)")
            
            formatted_lines.append("")
        
        return "\n".join(formatted_lines)
    
    def hybrid_retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        use_kg: bool = True
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        混合检索：向量检索 + 知识图谱增强
        
        Returns:
            (vector_results, kg_knowledge_text)
        """
        # 1. 向量检索（主要检索）
        vector_results = []
        if self.vector_available:
            query_embedding = self.embedding_generator.generate_embedding(query, is_query=True)
            
            try:
                vector_results = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k
                )
                if self.verbose:
                    print(f"📄 Vector search retrieved {len(vector_results)} relevant documents")
            except Exception as e:
                if self.verbose:
                    print(f"✗ Vector search failed: {e}")
                vector_results = []
        else:
            if self.verbose:
                print(f"⚠️  Vector search skipped (not available)")
        
        # 2. 知识图谱增强（可选）
        kg_knowledge = None
        if use_kg and self.kg_available:
            # 从查询中提取实体
            entities = self.extract_entities_from_query(query)
            
            # 从图谱查询相关知识
            if entities:
                kg_results = self.query_knowledge_graph(entities)
                kg_knowledge = self.format_kg_knowledge(kg_results)
        
        return vector_results, kg_knowledge
    
    def close(self):
        """关闭连接"""
        if self.kg_available:
            self.neo4j_manager.close()

