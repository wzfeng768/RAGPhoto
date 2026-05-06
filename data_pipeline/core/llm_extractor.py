"""
基于LLM的实体和关系抽取器
使用大语言模型智能抽取光电文献中的实体和关系
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from typing import Dict, List, Any, Optional, Tuple
import openai

from config.config import config
from .knowledge_graph import Entity, Relation, EntityType, RelationType

class LLMEntityRelationExtractor:
    """基于LLM的实体关系抽取器"""
    
    def __init__(self):
        """初始化LLM抽取器"""
        print("🤖 Initializing LLM Entity-Relation Extractor:")
        print(f"  Model: {config.llm_model}")
        print(f"  API URL: {config.openai_base_url}")
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # Entity extraction system prompt (English)
        self.entity_extraction_prompt = """
You are an AI assistant specialized in extracting entities from photovoltaic and solar cell literature.

Task: Identify and extract the following types of entities from the given text:

1. **Material**: Perovskite, silicon, organic materials, metal oxides, semiconductors, etc.
2. **Device**: Solar cells, LEDs, lasers, photodetectors, photovoltaic devices, etc.  
3. **Metric**: Efficiency, band gap, mobility, open-circuit voltage, short-circuit current, etc.
4. **Process**: Spin coating, evaporation, annealing, plasma treatment, fabrication methods, etc.
5. **Measurement**: CV, EQE, PL, XRD, SEM, UV-vis, characterization techniques, etc.
6. **Author**: Researcher names
7. **Institution**: Universities, research institutes, organizations
8. **Concept**: Energy levels, charge carriers, band gap engineering, physical concepts, etc.

Please return the extraction results in JSON format as follows:
```json
{{
  "entities": [
    {{
      "name": "Entity name",
      "type": "Entity type (Material/Device/Metric/etc.)",
      "description": "Brief description",
      "properties": {{"key": "value"}},
      "confidence": 0.8
    }}
  ]
}}
```

Important notes:
- Only extract entities that are explicitly mentioned, do not speculate
- Entity names should be standardized terms
- Confidence range: 0.5-1.0, higher values for more certain entities
- If no entities are found, return an empty list
- Use English names for entities when possible

Text content:
{text}
"""

        # Relation extraction system prompt (English)
        self.relation_extraction_prompt = """
You are an AI assistant specialized in extracting entity relationships from photovoltaic literature.

Task: Based on the given text and identified entities, extract the relationships between them.

Relation types:
1. **USES**: Usage relationship (e.g., solar cell uses perovskite material)
2. **HAS_PROPERTY**: Property relationship (e.g., material has efficiency value)
3. **FABRICATED_BY**: Fabrication relationship (e.g., material fabricated by spin coating)
4. **MEASURES**: Measurement relationship (e.g., EQE measures efficiency)
5. **IMPROVES**: Improvement relationship (e.g., process improves performance)
6. **CONTAINS**: Containment relationship (e.g., device contains material layer)
7. **RELATED_TO**: General relationship (e.g., material related to performance)

Entity list:
{entities}

Please return the relation extraction results in JSON format:
```json
{{
  "relations": [
    {{
      "source_entity": "Source entity name",
      "target_entity": "Target entity name", 
      "relation_type": "Relation type",
      "description": "Relation description",
      "properties": {{"context": "Related context"}},
      "confidence": 0.8
    }}
  ]
}}
```

Text content:
{text}
"""

    def extract_entities_from_text(self, text: str, paper_id: str = None) -> List[Entity]:
        """使用LLM从文本中抽取实体"""
        if not text or len(text.strip()) < 20:
            return []
        
        try:
            # 准备提示词
            prompt = self.entity_extraction_prompt.format(text=text)
            
            # 调用LLM
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的学术文献实体抽取专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度以获得更一致的结果
                max_tokens=2000
            )
            
            # 解析响应
            content = response.choices[0].message.content
            entities = self._parse_entity_response(content, paper_id)
            
            return entities
            
        except Exception as e:
            print(f"   ⚠️ LLM entity extraction failed: {e}")
            return []
    
    def extract_relations_from_text(self, text: str, entities: List[Entity], paper_id: str = None) -> List[Relation]:
        """使用LLM从文本中抽取关系"""
        if not text or not entities:
            return []
        
        try:
            # 准备实体列表
            entity_list = []
            for entity in entities:
                entity_list.append({
                    "name": entity.name,
                    "type": entity.type.value,
                    "id": entity.id
                })
            
            # 准备提示词
            prompt = self.relation_extraction_prompt.format(
                text=text,
                entities=json.dumps(entity_list, ensure_ascii=False, indent=2)
            )
            
            # 调用LLM
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的学术文献关系抽取专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # 解析响应
            content = response.choices[0].message.content
            relations = self._parse_relation_response(content, entities, paper_id)
            
            return relations
            
        except Exception as e:
            print(f"   ⚠️ LLM relation extraction failed: {e}")
            return []
    
    def _parse_entity_response(self, response_content: str, paper_id: str = None) -> List[Entity]:
        """解析LLM实体抽取响应"""
        entities = []
        
        try:
            # 尝试多种方式提取JSON
            json_str = None
            
            # 方法1: 提取```json```代码块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 方法2: 查找以{开头}结尾的JSON块
                json_match = re.search(r'\{[^{}]*"entities"[^{}]*\[[^\]]*\][^{}]*\}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # 方法3: 尝试直接解析整个响应
                    response_clean = response_content.strip()
                    if response_clean.startswith('{') and response_clean.endswith('}'):
                        json_str = response_clean
                    else:
                        print(f"   ⚠️ Cannot find valid JSON format, response content: {response_content[:200]}...")
                        return entities
            
            # 解析JSON
            data = json.loads(json_str)
            
            if 'entities' in data:
                for entity_data in data['entities']:
                    try:
                        # 验证实体类型
                        entity_type_str = entity_data.get('type', '')
                        entity_type = None
                        
                        for etype in EntityType:
                            if etype.value.lower() == entity_type_str.lower():
                                entity_type = etype
                                break
                        
                        if not entity_type:
                            continue
                        
                        # 创建实体
                        entity = Entity(
                            id=f"{entity_type.value}_{entity_data['name'].replace(' ', '_').replace('/', '_')}",
                            name=entity_data['name'],
                            type=entity_type,
                            description=entity_data.get('description', ''),
                            properties=entity_data.get('properties', {}),
                            source_papers=[paper_id] if paper_id else [],
                            confidence=float(entity_data.get('confidence', 0.8))
                        )
                        entities.append(entity)
                        
                    except Exception as e:
                        print(f"   ⚠️ Failed to parse entity: {e}")
                        continue
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parsing failed: {e}")
        except Exception as e:
            print(f"   ⚠️ Response parsing failed: {e}")
        
        return entities
    
    def _parse_relation_response(self, response_content: str, entities: List[Entity], paper_id: str = None) -> List[Relation]:
        """解析LLM关系抽取响应"""
        relations = []
        
        try:
            # 提取JSON部分
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if not json_match:
                json_str = response_content.strip()
            else:
                json_str = json_match.group(1)
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 创建实体名称到实体对象的映射
            entity_map = {entity.name.lower(): entity for entity in entities}
            
            if 'relations' in data:
                for relation_data in data['relations']:
                    try:
                        # 查找源实体和目标实体
                        source_name = relation_data.get('source_entity', '').lower()
                        target_name = relation_data.get('target_entity', '').lower()
                        
                        source_entity = entity_map.get(source_name)
                        target_entity = entity_map.get(target_name)
                        
                        if not source_entity or not target_entity:
                            continue
                        
                        # 验证关系类型
                        relation_type_str = relation_data.get('relation_type', '')
                        relation_type = None
                        
                        for rtype in RelationType:
                            if rtype.value.lower() == relation_type_str.lower():
                                relation_type = rtype
                                break
                        
                        if not relation_type:
                            continue
                        
                        # 创建关系
                        relation = Relation(
                            id=f"{relation_type.value}_{source_entity.id}_{target_entity.id}",
                            source_id=source_entity.id,
                            target_id=target_entity.id,
                            type=relation_type,
                            description=relation_data.get('description', ''),
                            properties=relation_data.get('properties', {}),
                            source_papers=[paper_id] if paper_id else [],
                            confidence=float(relation_data.get('confidence', 0.7))
                        )
                        relations.append(relation)
                        
                    except Exception as e:
                        print(f"   ⚠️ Failed to parse relation: {e}")
                        continue
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parsing failed: {e}")
        except Exception as e:
            print(f"   ⚠️ Response parsing failed: {e}")
        
        return relations
    
    def extract_entities_and_relations_batch(self, texts: List[str], paper_ids: List[str] = None) -> Tuple[List[Entity], List[Relation]]:
        """Batch extract entities and relations"""
        if paper_ids is None:
            paper_ids = [f"doc_{i}" for i in range(len(texts))]
        
        all_entities = []
        all_relations = []
        
        for i, (text, paper_id) in enumerate(zip(texts, paper_ids)):
            print(f"   🔍 LLM processing document {i+1}/{len(texts)}")
            
            # Extract entities
            entities = self.extract_entities_from_text(text, paper_id)
            all_entities.extend(entities)
            
            # Extract relations
            relations = self.extract_relations_from_text(text, entities, paper_id)
            all_relations.extend(relations)
        
        return all_entities, all_relations 