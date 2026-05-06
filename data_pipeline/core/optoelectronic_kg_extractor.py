"""
Traceable Optoelectronic Knowledge Graph Extractor
===================================================

Advanced LLM-based entity and relation extraction with full source traceability
for optoelectronic research literature.

Features:
- Domain-specific extraction for optoelectronic materials and devices
- Complete source traceability for every extracted entity and relation
- Enhanced confidence scoring based on evidence strength
- Support for figure/table reference tracking
- Context preservation for experimental conditions

Author: RAGPhoto Team
Date: 2025-10-22
Version: 2.0 - Traceable Optoelectronic Extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import openai

from config.config import config
from config.kg_extraction_prompts import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    RELATION_EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_CONFIG,
    validate_entity_response,
    validate_relation_response
)
from .knowledge_graph import Entity, Relation, EntityType, RelationType


@dataclass
class SourceTraceability:
    """Source traceability information for extracted knowledge"""
    text_snippet: str = ""
    context: str = ""
    figure_table_ref: str = ""
    section_hint: str = ""
    evidence_strength: str = "explicit"  # explicit, implicit, inferred
    chunk_id: str = ""
    paper_id: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for storage"""
        return {k: v for k, v in asdict(self).items() if v}


class TraceableEntity(Entity):
    """Entity with enhanced traceability information"""
    
    def __init__(self, *args, source_traceability: Optional[SourceTraceability] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_traceability = source_traceability or SourceTraceability()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including traceability"""
        base_dict = super().__dict__.copy()
        base_dict['source_traceability'] = self.source_traceability.to_dict()
        return base_dict


class TraceableRelation(Relation):
    """Relation with enhanced traceability information"""
    
    def __init__(self, *args, source_traceability: Optional[SourceTraceability] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_traceability = source_traceability or SourceTraceability()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including traceability"""
        base_dict = super().__dict__.copy()
        base_dict['source_traceability'] = self.source_traceability.to_dict()
        return base_dict


class OptoelectronicKGExtractor:
    """
    Advanced knowledge graph extractor for optoelectronic research literature
    with full source traceability.
    """
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean JSON string to handle common issues:
        - Invalid escape sequences (e.g., \\\\ -> \\)
        - Control characters
        - Unicode issues
        - Escaped braces ({{ -> {, }} -> })
        """
        try:
            # Remove control characters except newlines and tabs
            json_str = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', json_str)
            
            # Fix escaped braces in JSON structure (but preserve in string values)
            # Pattern: {{ or }} that are NOT inside string values
            # Strategy: Replace {{ with { and }} with } only when they're structural braces
            # We need to be careful not to replace braces inside string values
            def fix_escaped_braces(text):
                """Fix escaped braces {{ -> { and }} -> } in JSON structure"""
                result = []
                in_string = False
                escape_next = False
                i = 0
                
                while i < len(text):
                    char = text[i]
                    
                    if escape_next:
                        result.append(char)
                        escape_next = False
                        i += 1
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        result.append(char)
                        i += 1
                        continue
                    
                    if char == '"':
                        in_string = not in_string
                        result.append(char)
                        i += 1
                        continue
                    
                    if not in_string:
                        # Outside string: fix escaped braces
                        if i < len(text) - 1:
                            if text[i:i+2] == '{{':
                                result.append('{')
                                i += 2
                                continue
                            elif text[i:i+2] == '}}':
                                result.append('}')
                                i += 2
                                continue
                    
                    result.append(char)
                    i += 1
                
                return ''.join(result)
            
            json_str = fix_escaped_braces(json_str)
            
            # Fix invalid escape sequences in string values
            # Pattern: match string values (content between quotes)
            def fix_escapes_in_string(match):
                content = match.group(1)
                
                # Step 1: Fix three or more consecutive backslashes (\\\ -> \\)
                # This handles cases like $\\\mathrm{...}$ -> $\\mathrm{...}$
                # Strategy: replace odd sequences (3, 5, 7, ...) with even pairs
                # 3 backslashes -> 2, 5 -> 4, 7 -> 6, etc.
                # Even sequences (4, 6, 8, ...) are kept as is
                def fix_backslashes(m):
                    count = len(m.group(0))
                    if count % 2 == 1:
                        # Odd: reduce by 1 to make it even (3->2, 5->4, 7->6)
                        return '\\\\' * ((count - 1) // 2)
                    else:
                        # Even: keep as pairs (4->4, 6->6, 8->8)
                        return '\\\\' * (count // 2)
                content = re.sub(r'\\{3,}', fix_backslashes, content)
                
                # Step 2: Fix invalid escape sequences (but preserve valid ones)
                # Valid escapes: \n, \t, \r, \b, \f, \", \\, \/, \uXXXX
                # Invalid escapes like \m, \a, etc. should be escaped as \\m, \\a
                # But we need to be careful not to break valid sequences
                def fix_invalid_escape(m):
                    seq = m.group(0)
                    # If it's a valid escape sequence, keep it
                    if seq in ['\\n', '\\t', '\\r', '\\b', '\\f', '\\"', '\\\\', '\\/']:
                        return seq
                    # If it's a Unicode escape \uXXXX, keep it
                    if seq.startswith('\\u') and len(seq) == 6:
                        return seq
                    # Otherwise, escape the backslash: \m -> \\m
                    return '\\\\' + seq[1:]
                
                # Match backslash followed by a character
                content = re.sub(r'\\.', fix_invalid_escape, content)
                
                return f'"{content}"'
            
            # Match string values (content between quotes)
            # This regex matches: "..." where ... can contain escaped quotes
            json_str = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', fix_escapes_in_string, json_str)
            
            return json_str.strip()
        except Exception as e:
            print(f"      ⚠️ Error cleaning JSON string: {e}")
            return json_str  # Return original if cleaning fails
    
    def __init__(self, enable_traceability: bool = True):
        """
        Initialize the optoelectronic KG extractor
        
        Args:
            enable_traceability: Whether to enforce traceability requirements
        """
        print("🔬 Initializing Traceable Optoelectronic KG Extractor:")
        print(f"  Model: {config.llm_model}")
        print(f"  API URL: {config.openai_base_url}")
        print(f"  Traceability: {'✓ Enabled' if enable_traceability else '✗ Disabled'}")
        
        self.enable_traceability = enable_traceability
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # Load extraction configuration
        self.entity_config = EXTRACTION_CONFIG["entity_extraction"]
        self.relation_config = EXTRACTION_CONFIG["relation_extraction"]
        self.traceability_config = EXTRACTION_CONFIG["traceability"]
        
        # Statistics tracking
        self.stats = {
            "total_entities": 0,
            "total_relations": 0,
            "entities_with_traceability": 0,
            "relations_with_traceability": 0,
            "extraction_failures": 0,
            "reference_chunks_filtered": 0
        }
    
    def _is_reference_content(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        检测文本是否主要包含参考文献内容
        
        Args:
            text: 要检测的文本
            metadata: 文本元数据，可能包含section_type信息
        
        Returns:
            如果文本主要是参考文献内容则返回True
        """
        # 首先检查元数据中的section_type
        if metadata and metadata.get('section_type') == 'references':
            return True
        
        text_lower = text.lower().strip()
        
        # 如果文本太短，不太可能是参考文献
        if len(text_lower) < 50:
            return False
        
        # 检查常见的参考文献标识符模式
        reference_patterns = [
            # 引用编号模式
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d{4}\)',  # (2023), (2022), etc.
            r'\d+\.\s+\w+.*?\d{4}',  # 1. Author ... 2023
            # DOI模式
            r'doi:', r'DOI:', r'https?://doi\.org/',
            # 期刊名称模式  
            r'journal|proceedings|conference|nature|science|phys\.|chem\.|j\.',
            # 作者名模式
            r'\w+,\s*\w+\.\s*\w*\.?',  # Smith, J. A.
            # 引用关键词
            r'et\s+al\.', r'vol\.', r'pp\.', r'no\.', r'issue',
        ]
        
        # 计算匹配的模式数量
        matches = 0
        for pattern in reference_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        
        # 计算引用特征密度
        total_patterns = len(reference_patterns)
        reference_density = matches / total_patterns
        
        # 检查是否有大量数字引用格式
        citation_brackets = len(re.findall(r'\[\d+\]', text))
        citation_parens = len(re.findall(r'\(\d{4}\)', text))
        total_citations = citation_brackets + citation_parens
        
        # 如果引用密度高或者有大量引用格式，认为是参考文献内容
        if reference_density > 0.3 or total_citations > 3:
            return True
        
        # 检查是否包含连续的引用模式
        lines = text.split('\n')
        reference_lines = 0
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # 忽略太短的行
                # 检查该行是否像引用
                if (re.search(r'^\d+\.', line) or 
                    re.search(r'^\[\d+\]', line) or
                    re.search(r'\(\d{4}\)', line) and re.search(r'\w+,\s*\w+', line)):
                    reference_lines += 1
        
        # 如果大部分行都像引用，认为是参考文献内容
        if len(lines) > 5 and reference_lines / len(lines) > 0.5:
            return True
            
        return False
    
    def extract_entities_from_text(
        self, 
        text: str, 
        paper_id: str = None,
        chunk_id: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TraceableEntity]:
        """
        Extract entities with traceability from research text
        
        Args:
            text: Text to extract entities from
            paper_id: Identifier for the source paper
            chunk_id: Identifier for the text chunk
            metadata: Additional metadata (section, page, etc.)
        
        Returns:
            List of traceable entities
        """
        if not text or len(text.strip()) < self.entity_config["min_text_length"]:
            return []
        
        # 检查是否为参考文献内容，如果是则跳过提取
        if self._is_reference_content(text, metadata):
            self.stats["reference_chunks_filtered"] += 1
            print(f"      🚫 跳过参考文献内容 (chunk_id: {chunk_id or 'unknown'})")
            return []
        
        try:
            # Prepare the prompt - 使用字符串模板来避免.format()的大括号问题
            from string import Template
            # 将 {text} 替换为 $text 以便使用 Template
            template_str = ENTITY_EXTRACTION_PROMPT.replace('{text}', '$text')
            template = Template(template_str)
            prompt = template.safe_substitute(text=text)
            
            # Call LLM with optimized parameters
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.entity_config["temperature"],
                max_tokens=self.entity_config["max_tokens"],
                top_p=self.entity_config["top_p"]
            )
            
            # Parse response
            content = response.choices[0].message.content
            entities = self._parse_entity_response(
                content, 
                paper_id=paper_id,
                chunk_id=chunk_id,
                metadata=metadata
            )
            
            # Update statistics
            self.stats["total_entities"] += len(entities)
            self.stats["entities_with_traceability"] += sum(
                1 for e in entities if e.source_traceability.text_snippet
            )
            
            print(f"      🎯 Extracted {len(entities)} traceable entities")
            if self.enable_traceability:
                traced = sum(1 for e in entities if e.source_traceability.text_snippet)
                print(f"         ✓ {traced}/{len(entities)} with full traceability")
            
            return entities
            
        except Exception as e:
            print(f"   ⚠️ Entity extraction failed: {e}")
            self.stats["extraction_failures"] += 1
            return []
    
    def extract_relations_from_text(
        self,
        text: str,
        entities: List[TraceableEntity],
        paper_id: str = None,
        chunk_id: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TraceableRelation]:
        """
        Extract relations with traceability between entities
        
        Args:
            text: Source text
            entities: List of entities to find relations between
            paper_id: Identifier for the source paper
            chunk_id: Identifier for the text chunk
            metadata: Additional metadata
        
        Returns:
            List of traceable relations
        """
        if not text or not entities or len(entities) < self.relation_config["min_entities"]:
            return []
        
        try:
            # Prepare entity list for the prompt
            entity_list = self._format_entity_list_for_prompt(entities)
            
            # Prepare the prompt - 使用字符串模板来避免.format()的大括号问题
            from string import Template
            # 将占位符替换为 $ 格式以便使用 Template
            template_str = RELATION_EXTRACTION_PROMPT.replace('{entities}', '$entities').replace('{text}', '$text')
            template = Template(template_str)
            prompt = template.safe_substitute(entities=entity_list, text=text)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": RELATION_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.relation_config["temperature"],
                max_tokens=self.relation_config["max_tokens"],
                top_p=self.relation_config["top_p"]
            )
            
            # Parse response
            content = response.choices[0].message.content
            relations = self._parse_relation_response(
                content,
                entities,
                paper_id=paper_id,
                chunk_id=chunk_id,
                metadata=metadata
            )
            
            # Update statistics
            self.stats["total_relations"] += len(relations)
            self.stats["relations_with_traceability"] += sum(
                1 for r in relations if r.source_traceability.text_snippet
            )
            
            print(f"      🔗 Extracted {len(relations)} traceable relations")
            if self.enable_traceability:
                traced = sum(1 for r in relations if r.source_traceability.text_snippet)
                print(f"         ✓ {traced}/{len(relations)} with full traceability")
            
            return relations
            
        except Exception as e:
            print(f"   ⚠️ Relation extraction failed: {e}")
            self.stats["extraction_failures"] += 1
            return []
    
    def _format_entity_list_for_prompt(self, entities: List[TraceableEntity]) -> str:
        """Format entity list for relation extraction prompt"""
        lines = []
        for entity in entities:
            line = f"- **{entity.name}** ({entity.type.value})"
            
            # Add properties if available
            if entity.properties:
                props = []
                for key, value in entity.properties.items():
                    if value and key not in ['source_traceability']:
                        props.append(f"{key}: {value}")
                if props:
                    line += f" [{', '.join(props)}]"
            
            # Add description if available
            if entity.description:
                line += f"\n  Description: {entity.description[:100]}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _parse_entity_response(
        self,
        content: str,
        paper_id: str = None,
        chunk_id: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TraceableEntity]:
        """Parse LLM entity response with traceability validation"""
        entities = []
        
        try:
            # Check if response is plain text explanation (not JSON)
            # Common patterns: "no entities", "does not contain", "cannot extract"
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in [
                'no entities', 'does not contain', 'cannot extract', 
                'no scientific content', 'bibliographic references only',
                'consists solely of', 'does not have any'
            ]) and not ('{' in content and '}' in content):
                # LLM returned plain text explanation instead of JSON
                # This is acceptable - just return empty entities
                print(f"      ℹ️  LLM indicated no entities in this text (acceptable)")
                return entities  # Return empty list
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                # Check if content looks like JSON (starts with { or [)
                content_stripped = content.strip()
                if content_stripped.startswith('{') or content_stripped.startswith('['):
                    json_str = content_stripped
                else:
                    # Not JSON format - likely plain text explanation
                    print(f"      ℹ️  Response is not JSON format, assuming no entities")
                    return entities  # Return empty list
            
            # Clean up JSON string before parsing
            json_str = self._clean_json_string(json_str)
            
            # Parse JSON with cleanup
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Clean up common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                json_str = self._clean_json_string(json_str)  # Clean again
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    # Final attempt failed - check if it's a plain text explanation
                    if len(json_str) > 100 and not ('{' in json_str and '}' in json_str):
                        print(f"      ℹ️  LLM returned explanation text instead of JSON (acceptable)")
                    else:
                        print(f"      ⚠️ JSON parsing error after cleanup: {e2}")
                        print(f"      ⚠️ Problematic JSON (first 500 chars): {json_str[:500]}")
                    return entities  # Return empty or partial results
            
            # Process entities
            for entity_data in data.get('entities', []):
                try:
                    # Validate response format
                    if self.enable_traceability and not validate_entity_response(entity_data):
                        print(f"      ⚠️ Entity missing traceability: {entity_data.get('name', 'unknown')}")
                        continue
                    
                    # Get entity type
                    entity_type_str = entity_data.get('type', '').strip()
                    entity_type = self._map_entity_type(entity_type_str)
                    entity_name = entity_data['name'].strip()
                    
                    # Generate unique ID based on name and type (for entity merging)
                    entity_id = Entity.generate_id(entity_name, entity_type)
                    
                    # Extract traceability information
                    traceability_data = entity_data.get('source_traceability', {})
                    source_traceability = SourceTraceability(
                        text_snippet=traceability_data.get('text_snippet', '')[:self.traceability_config["max_snippet_length"]],
                        context=traceability_data.get('context', ''),
                        figure_table_ref=traceability_data.get('figure_table_ref', ''),
                        section_hint=traceability_data.get('section_hint', ''),
                        chunk_id=chunk_id or '',
                        paper_id=paper_id or ''
                    )
                    
                    # Get confidence score
                    confidence = float(entity_data.get('confidence', 0.8))
                    
                    # Skip low-confidence entities
                    if confidence < self.entity_config["min_confidence"]:
                        continue
                    
                    # Create traceable entity
                    entity = TraceableEntity(
                        id=entity_id,
                        name=entity_name,
                        type=entity_type,
                        description=entity_data.get('description', ''),
                        properties=entity_data.get('properties', {}),
                        confidence=confidence,
                        source_papers=[paper_id] if paper_id else [],
                        source_traceability=source_traceability
                    )
                    
                    entities.append(entity)
                    
                except Exception as e:
                    print(f"      ⚠️ Error processing entity: {e}")
                    continue
            
        except Exception as e:
            print(f"   ⚠️ Entity response parsing failed: {e}")
        
        return entities
    
    def _parse_relation_response(
        self,
        content: str,
        entities: List[TraceableEntity],
        paper_id: str = None,
        chunk_id: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TraceableRelation]:
        """Parse LLM relation response with traceability validation"""
        relations = []
        entity_name_to_entity = {entity.name: entity for entity in entities}
        
        try:
            # Check if response is plain text explanation (not JSON)
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in [
                'no relations', 'no relationships', 'does not contain',
                'cannot extract', 'no entities', 'no scientific content'
            ]) and not ('{' in content and '}' in content):
                # LLM returned plain text explanation instead of JSON
                print(f"      ℹ️  LLM indicated no relations in this text (acceptable)")
                return relations  # Return empty list
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                content_stripped = content.strip()
                if content_stripped.startswith('{') or content_stripped.startswith('['):
                    json_str = content_stripped
                else:
                    # Not JSON format - likely plain text explanation
                    print(f"      ℹ️  Response is not JSON format, assuming no relations")
                    return relations  # Return empty list
            
            # Clean up JSON string before parsing
            json_str = self._clean_json_string(json_str)
            
            # Parse JSON with cleanup
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = self._clean_json_string(json_str)  # Clean again
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    # Final attempt failed - check if it's a plain text explanation
                    if len(json_str) > 100 and not ('{' in json_str and '}' in json_str):
                        print(f"      ℹ️  LLM returned explanation text instead of JSON (acceptable)")
                    else:
                        print(f"      ⚠️ JSON parsing error after cleanup: {e2}")
                        print(f"      ⚠️ Problematic JSON (first 500 chars): {json_str[:500]}")
                    return relations  # Return empty or partial results
            
            # Process relations
            for relation_data in data.get('relations', []):
                try:
                    # Validate response format
                    if self.enable_traceability and not validate_relation_response(relation_data):
                        print(f"      ⚠️ Relation missing traceability")
                        continue
                    
                    # Get source and target entities
                    source_name = relation_data.get('source_entity', '').strip()
                    target_name = relation_data.get('target_entity', '').strip()
                    
                    # Validate entity existence
                    if source_name not in entity_name_to_entity or target_name not in entity_name_to_entity:
                        continue
                    
                    source_entity = entity_name_to_entity[source_name]
                    target_entity = entity_name_to_entity[target_name]
                    
                    # Get relation type
                    relation_type_str = relation_data.get('relation_type', 'RELATED_TO').upper()
                    try:
                        relation_type = RelationType(relation_type_str)
                    except ValueError:
                        relation_type = RelationType.RELATED_TO  # Default fallback
                    
                    # Extract traceability information
                    traceability_data = relation_data.get('source_traceability', {})
                    source_traceability = SourceTraceability(
                        text_snippet=traceability_data.get('text_snippet', '')[:self.traceability_config["max_snippet_length"]],
                        context=traceability_data.get('context', ''),
                        figure_table_ref=traceability_data.get('figure_table_ref', ''),
                        evidence_strength=traceability_data.get('evidence_strength', 'explicit'),
                        chunk_id=chunk_id or '',
                        paper_id=paper_id or ''
                    )
                    
                    # Get confidence score
                    confidence = float(relation_data.get('confidence', 0.8))
                    
                    # Skip low-confidence relations
                    if confidence < self.relation_config["min_confidence"]:
                        continue
                    
                    # Create traceable relation
                    relation = TraceableRelation(
                        id=f"{paper_id}_{chunk_id}_{len(relations)}" if paper_id and chunk_id else f"relation_{len(relations)}",
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        type=relation_type,
                        description=relation_data.get('description', ''),
                        properties=relation_data.get('properties', {}),
                        confidence=confidence,
                        source_papers=[paper_id] if paper_id else [],
                        source_traceability=source_traceability
                    )
                    
                    relations.append(relation)
                    
                except Exception as e:
                    print(f"      ⚠️ Error processing relation: {e}")
                    continue
            
        except Exception as e:
            print(f"   ⚠️ Relation response parsing failed: {e}")
        
        return relations
    
    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map type string to EntityType enum"""
        type_str = type_str.upper().strip()
        
        # Direct mapping
        type_mapping = {
            'MATERIAL': EntityType.MATERIAL,
            'DEVICE': EntityType.DEVICE,
            'METRIC': EntityType.METRIC,
            'PROCESS': EntityType.PROCESS,
            'MEASUREMENT': EntityType.MEASUREMENT,
            'PARAMETER': EntityType.PARAMETER,
            'CONCEPT': EntityType.CONCEPT,
            'AUTHOR': EntityType.AUTHOR,
            'INSTITUTION': EntityType.INSTITUTION,
            'PAPER': EntityType.PAPER,
            'FORMULA': EntityType.FORMULA,
            'FORM': EntityType.FORMULA,
            'EQUATION': EntityType.FORMULA,
            'EQ': EntityType.FORMULA
        }
        
        return type_mapping.get(type_str, EntityType.CONCEPT)
    
    def extract_entities_and_relations_batch(
        self,
        text_chunks: List[str],
        paper_id: str = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[TraceableEntity], List[TraceableRelation]]:
        """
        Batch extraction with traceability for multiple text chunks
        
        Args:
            text_chunks: List of text chunks to process
            paper_id: Identifier for the source paper
            metadata_list: List of metadata dicts for each chunk
        
        Returns:
            Tuple of (entities, relations) with full traceability
        """
        all_entities = []
        all_relations = []
        
        if metadata_list is None:
            metadata_list = [None] * len(text_chunks)
        
        print(f"   🔬 Traceable batch processing: {len(text_chunks)} chunks")
        
        for i, (chunk, metadata) in enumerate(zip(text_chunks, metadata_list)):
            chunk_id = f"chunk_{i}"
            print(f"      Processing chunk {i+1}/{len(text_chunks)}...")
            
            # Extract entities
            entities = self.extract_entities_from_text(
                chunk,
                paper_id=paper_id,
                chunk_id=chunk_id,
                metadata=metadata
            )
            all_entities.extend(entities)
            
            # Extract relations
            if entities:
                relations = self.extract_relations_from_text(
                    chunk,
                    entities,
                    paper_id=paper_id,
                    chunk_id=chunk_id,
                    metadata=metadata
                )
                all_relations.extend(relations)
        
        print(f"   ✅ Traceable batch complete:")
        print(f"      - {len(all_entities)} entities")
        print(f"      - {len(all_relations)} relations")
        
        if self.enable_traceability:
            traced_entities = sum(1 for e in all_entities if e.source_traceability.text_snippet)
            traced_relations = sum(1 for r in all_relations if r.source_traceability.text_snippet)
            print(f"      - {traced_entities}/{len(all_entities)} entities with traceability")
            print(f"      - {traced_relations}/{len(all_relations)} relations with traceability")
        
        return all_entities, all_relations
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics including traceability coverage"""
        total_entities = self.stats["total_entities"]
        total_relations = self.stats["total_relations"]
        
        return {
            "total_entities": total_entities,
            "total_relations": total_relations,
            "entities_with_traceability": self.stats["entities_with_traceability"],
            "relations_with_traceability": self.stats["relations_with_traceability"],
            "entity_traceability_rate": (
                self.stats["entities_with_traceability"] / total_entities * 100
                if total_entities > 0 else 0
            ),
            "relation_traceability_rate": (
                self.stats["relations_with_traceability"] / total_relations * 100
                if total_relations > 0 else 0
            ),
            "extraction_failures": self.stats["extraction_failures"],
            "reference_chunks_filtered": self.stats["reference_chunks_filtered"]
        }
    
    def print_statistics(self):
        """Print extraction statistics"""
        stats = self.get_extraction_statistics()
        print("\n" + "="*70)
        print("📊 EXTRACTION STATISTICS")
        print("="*70)
        print(f"Total Entities Extracted:        {stats['total_entities']}")
        print(f"  - With Traceability:           {stats['entities_with_traceability']} ({stats['entity_traceability_rate']:.1f}%)")
        print(f"Total Relations Extracted:       {stats['total_relations']}")
        print(f"  - With Traceability:           {stats['relations_with_traceability']} ({stats['relation_traceability_rate']:.1f}%)")
        print(f"Reference Chunks Filtered:       {stats['reference_chunks_filtered']}")
        print(f"Extraction Failures:             {stats['extraction_failures']}")
        print("="*70 + "\n")

